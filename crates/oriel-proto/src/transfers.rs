//! Process-wide table of file transfers in progress, for showing them to the user.
//!
//! A transfer is one file of an offer moving over one connection, in one direction: read
//! through this process's transfer mount ([`TransferDirection::Incoming`]) or served to the
//! peer ([`TransferDirection::Outgoing`]). Every reader of the same file (a copy plus a
//! thumbnailer, a reopened stream after a seek) adds to the same entry, which ends when the
//! last of them does. A transfer can be cancelled: its reads then fail, and so do later
//! reads of the same file of the same offer (readers reopen streams that fail).
//!
//! A transfer can also be paused. Only the reading side can pause one without the read
//! failing, so a paused [`TransferDirection::Incoming`] transfer holds its readers here,
//! and pausing an [`TransferDirection::Outgoing`] one queues a request for the peer to hold
//! its readers (see [`take_peer_requests`]). A pause lasts while the transfer runs: when a
//! paused incoming transfer's last reader is done, the pause is forgotten (nothing would
//! be listed to resume it from), and when a paused outgoing one ends here (all of the file
//! sent, or the peer closed it) the peer is asked to resume it, after [`RELEASE_GRACE`]:
//! a reader that reopens its stream (a seek) within it continues the paused transfer. Cancelling a paused
//! outgoing transfer asks the peer to resume it too, so its held reads go on and fail.
//!
//! Entries are keyed by `quinn::Connection::stable_id`, an address a later connection can
//! reuse. Cancels, pauses and peer requests are only remembered for a connection while a
//! [`ConnectionGuard`] for it is alive; dropping the guard forgets them. A transfer that
//! ends later (its stream outlived the guard) records nothing, and cannot outlive the
//! connection's address: its stream or reader keeps the connection allocated.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

/// Transfers younger than this are not listed: file managers and thumbnailers open files
/// briefly to sniff their type or render a preview. Nor are those that have moved nothing
/// (a file held open without reading it), unless paused or cancelled.
const MIN_LISTED_AGE: Duration = Duration::from_secs(1);
/// How many cancelled files are remembered, so reopening them fails too.
const REMEMBERED_CANCELS: usize = 256;
/// How many paused files are remembered per connection, so a reader opening one stays
/// paused.
const REMEMBERED_PAUSES: usize = 256;
/// How long a paused outgoing transfer that ended stays paused for a new stream of the
/// same file before the peer is asked to resume it.
const RELEASE_GRACE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// The peer's file, read by an app on this side.
    Incoming,
    /// This side's file, read by an app on the peer.
    Outgoing,
}

/// A snapshot of one transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferInfo {
    pub id: u64,
    pub direction: TransferDirection,
    /// The file's name.
    pub name: String,
    pub size: u64,
    /// How far into the file the reader has got.
    pub position: u64,
    /// Bytes moved over the connection so far (re-reads count again).
    pub moved: u64,
    /// How long ago the transfer started.
    pub age: Duration,
    pub cancelled: bool,
    pub paused: bool,
}

/// Which file of which offer a transfer reads, as the peer knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferFile {
    pub direction: TransferDirection,
    pub offer_id: u64,
    pub path: Vec<String>,
}

/// Pause (`paused`) or resume the peer's readers of an outgoing transfer's file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerRequest {
    pub offer_id: u64,
    pub path: Vec<String>,
    pub paused: bool,
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct Key {
    conn: usize,
    direction: TransferDirection,
    offer_id: u64,
    path: Vec<String>,
}

struct Entry {
    id: u64,
    size: AtomicU64,
    started: Instant,
    position: AtomicU64,
    moved: AtomicU64,
    cancel: tokio::sync::watch::Sender<bool>,
    paused: tokio::sync::watch::Sender<bool>,
}

struct Slot {
    entry: Arc<Entry>,
    readers: usize,
}

struct Table {
    slots: HashMap<Key, Slot>,
    /// Connections with a live [`ConnectionGuard`] (counted: guards may overlap).
    live: HashMap<usize, usize>,
    cancelled: VecDeque<Key>,
    paused: HashSet<Key>,
    /// Requests for peers, in order.
    requests: Vec<(usize, PeerRequest)>,
    /// Paused outgoing transfers that ended, waiting out [`RELEASE_GRACE`], by generation.
    lingering: HashMap<Key, u64>,
    next_linger: u64,
    next_id: u64,
}

impl Table {
    fn queue_request(&mut self, key: &Key, paused: bool) {
        if !self.live.contains_key(&key.conn) {
            return;
        }
        self.requests.push((
            key.conn,
            PeerRequest {
                offer_id: key.offer_id,
                path: key.path.clone(),
                paused,
            },
        ));
        REQUESTS_QUEUED.send_modify(|()| {});
    }

    /// Paused outgoing transfer `key` ended: ask the peer to resume it unless a new stream
    /// of the file arrives within [`RELEASE_GRACE`] (see [`track`]).
    fn linger(&mut self, key: &Key) {
        if !self.live.contains_key(&key.conn) {
            return;
        }
        let Ok(rt) = tokio::runtime::Handle::try_current() else {
            self.queue_request(key, false);
            return;
        };
        let generation = self.next_linger;
        self.next_linger += 1;
        self.lingering.insert(key.clone(), generation);
        let key = key.clone();
        rt.spawn(async move {
            tokio::time::sleep(RELEASE_GRACE).await;
            let mut table = TABLE.lock().unwrap();
            if table.lingering.get(&key) == Some(&generation) {
                table.lingering.remove(&key);
                table.queue_request(&key, false);
            }
        });
    }
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(|| {
    Mutex::new(Table {
        slots: HashMap::new(),
        live: HashMap::new(),
        cancelled: VecDeque::new(),
        paused: HashSet::new(),
        requests: Vec::new(),
        lingering: HashMap::new(),
        next_linger: 0,
        next_id: 1,
    })
});

/// Changes whenever a request for a peer is queued.
static REQUESTS_QUEUED: LazyLock<tokio::sync::watch::Sender<()>> =
    LazyLock::new(|| tokio::sync::watch::channel(()).0);

/// One reader's part in a transfer; the transfer ends when the last one is dropped.
pub struct TransferGuard {
    key: Key,
    entry: Arc<Entry>,
}

impl std::fmt::Debug for TransferGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransferGuard")
            .field("id", &self.entry.id)
            .finish()
    }
}

/// Start reading file `path` of offer `offer_id` (`size` bytes) over connection `conn`
/// (`quinn::Connection::stable_id`), joining the transfer of that file if one is running.
pub fn track(
    conn: usize,
    direction: TransferDirection,
    offer_id: u64,
    path: &[String],
    size: u64,
) -> TransferGuard {
    let key = Key {
        conn,
        direction,
        offer_id,
        path: path.to_vec(),
    };
    let mut table = TABLE.lock().unwrap();
    let next_id = table.next_id;
    let was_cancelled = table.cancelled.contains(&key);
    // A new stream of a paused outgoing file that just ended continues its pause.
    let lingered = table.lingering.remove(&key).is_some();
    let was_paused = lingered || table.paused.contains(&key);
    let slot = table.slots.entry(key.clone()).or_insert_with(|| Slot {
        entry: Arc::new(Entry {
            id: next_id,
            size: AtomicU64::new(size),
            started: Instant::now(),
            position: AtomicU64::new(0),
            moved: AtomicU64::new(0),
            cancel: tokio::sync::watch::channel(was_cancelled).0,
            paused: tokio::sync::watch::channel(was_paused).0,
        }),
        readers: 0,
    });
    slot.readers += 1;
    let entry = Arc::clone(&slot.entry);
    // A reader that joins may have found the file grown.
    entry.size.fetch_max(size, Ordering::Relaxed);
    if entry.id == next_id {
        table.next_id += 1;
        if lingered {
            // The peer's reader may have closed and reopened the file, forgetting the pause.
            table.queue_request(&key, true);
        }
    } else if direction == TransferDirection::Outgoing
        && *entry.paused.borrow()
        && !*entry.cancel.borrow()
    {
        // A new stream from the peer joins a paused transfer: its reader may have closed
        // and reopened the file, forgetting the pause there. Ask again.
        table.queue_request(&key, true);
    }
    TransferGuard { key, entry }
}

impl TransferGuard {
    /// The reader got to `position` in the file, moving `moved` more bytes over the
    /// connection.
    pub fn advance(&self, position: u64, moved: u64) {
        self.entry.position.fetch_max(position, Ordering::Relaxed);
        self.entry.moved.fetch_add(moved, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.entry.cancel.borrow()
    }

    pub fn is_paused(&self) -> bool {
        *self.entry.paused.borrow()
    }

    /// Resolves once the transfer is not paused (at once if it is not), or is cancelled.
    pub async fn unpaused(&self) {
        let mut paused = self.entry.paused.subscribe();
        let mut cancel = self.entry.cancel.subscribe();
        loop {
            if !*paused.borrow_and_update() || *cancel.borrow_and_update() {
                return;
            }
            // Both senders live in the entry this guard keeps alive.
            tokio::select! {
                _ = paused.changed() => {}
                _ = cancel.changed() => {}
            }
        }
    }

    /// Resolves once the transfer is cancelled (`true`) or has ended (`false`; never while
    /// this guard is alive).
    pub fn cancelled(&self) -> impl std::future::Future<Output = bool> + Send + 'static {
        let mut rx = self.entry.cancel.subscribe();
        async move { rx.wait_for(|c| *c).await.is_ok() }
    }
}

impl Drop for TransferGuard {
    fn drop(&mut self) {
        let mut table = TABLE.lock().unwrap();
        let Some(slot) = table.slots.get_mut(&self.key) else {
            return;
        };
        slot.readers -= 1;
        if slot.readers > 0 {
            return;
        }
        table.slots.remove(&self.key);
        let e = &self.entry;
        match self.key.direction {
            // A pause ends with the transfer: nothing would be listed to resume it from.
            TransferDirection::Incoming => {
                table.paused.remove(&self.key);
            }
            // The peer still holds its readers, waiting for a resume nobody can ask for:
            // nothing is left to hold back. (Cancelling already resumed them.)
            TransferDirection::Outgoing => {
                if *e.paused.borrow() && !*e.cancel.borrow() {
                    table.linger(&self.key);
                }
            }
        }
    }
}

/// The transfers over connection `conn` that have run long enough to show, oldest first.
pub fn list(conn: usize) -> Vec<TransferInfo> {
    let table = TABLE.lock().unwrap();
    let mut out: Vec<(Instant, TransferInfo)> = table
        .slots
        .iter()
        .filter(|(key, slot)| {
            let e = &slot.entry;
            key.conn == conn
                && e.started.elapsed() >= MIN_LISTED_AGE
                && (e.moved.load(Ordering::Relaxed) > 0 || *e.paused.borrow() || *e.cancel.borrow())
        })
        .map(|(key, slot)| {
            let e = &slot.entry;
            let size = e.size.load(Ordering::Relaxed);
            (
                e.started,
                TransferInfo {
                    id: e.id,
                    direction: key.direction,
                    name: key.path.last().cloned().unwrap_or_default(),
                    size,
                    position: e.position.load(Ordering::Relaxed).min(size),
                    moved: e.moved.load(Ordering::Relaxed),
                    age: e.started.elapsed(),
                    cancelled: *e.cancel.borrow(),
                    // A cancelled transfer's readers are let go whether paused or not.
                    paused: *e.paused.borrow() && !*e.cancel.borrow(),
                },
            )
        })
        .collect();
    out.sort_by_key(|(started, info)| (*started, info.id));
    out.into_iter().map(|(_, info)| info).collect()
}

/// Cancel transfer `id`: its readers' pending and later reads fail, and so does reading
/// that file of that offer again. A paused outgoing one is resumed on the peer, so the
/// reads held there go on and fail. `false` if it ended.
pub fn cancel(id: u64) -> bool {
    let mut table = TABLE.lock().unwrap();
    let Some((key, slot)) = table.slots.iter().find(|(_, slot)| slot.entry.id == id) else {
        return false;
    };
    let key = key.clone();
    let entry = Arc::clone(&slot.entry);
    entry.cancel.send_replace(true);
    if key.direction == TransferDirection::Outgoing && entry.paused.send_replace(false) {
        table.queue_request(&key, false);
    }
    if table.live.contains_key(&key.conn) && !table.cancelled.contains(&key) {
        if table.cancelled.len() == REMEMBERED_CANCELS {
            table.cancelled.pop_front();
        }
        table.cancelled.push_back(key);
    }
    true
}

/// Pause or resume transfer `id`. An incoming transfer's readers here wait while it is
/// paused; for an outgoing one, the peer is asked to hold its readers (see
/// [`take_peer_requests`]). `false` if the transfer ended or was cancelled.
pub fn set_paused(id: u64, paused: bool) -> bool {
    let mut table = TABLE.lock().unwrap();
    let Some((key, slot)) = table.slots.iter().find(|(_, slot)| slot.entry.id == id) else {
        return false;
    };
    if *slot.entry.cancel.borrow() {
        // Its readers are being let go; a pause would hold them again, for good.
        return false;
    }
    let key = key.clone();
    slot.entry.paused.send_replace(paused);
    match key.direction {
        TransferDirection::Incoming => remember_pause(&mut table, &key, paused),
        TransferDirection::Outgoing => table.queue_request(&key, paused),
    }
    true
}

/// Pause or resume the readers here of `file` over connection `conn` (an incoming
/// transfer, at the peer's request): those reading it now, and any that open it while
/// it stays paused.
pub fn set_paused_file(conn: usize, file: &TransferFile, paused: bool) {
    let key = Key {
        conn,
        direction: file.direction,
        offer_id: file.offer_id,
        path: file.path.clone(),
    };
    let mut table = TABLE.lock().unwrap();
    if let Some(slot) = table.slots.get(&key) {
        if paused && *slot.entry.cancel.borrow() {
            return;
        }
        slot.entry.paused.send_replace(paused);
    }
    remember_pause(&mut table, &key, paused);
}

fn remember_pause(table: &mut Table, key: &Key, paused: bool) {
    if !paused {
        table.paused.remove(key);
        return;
    }
    if !table.live.contains_key(&key.conn) {
        return;
    }
    if table.paused.contains(key) {
        return;
    }
    // Capped per connection, so one peer's pauses cannot push out another's.
    let mine = table.paused.iter().filter(|k| k.conn == key.conn);
    if mine.clone().count() >= REMEMBERED_PAUSES {
        // Forget one that is no longer running; a running one can still be resumed.
        let idle = mine
            .clone()
            .find(|k| !table.slots.contains_key(*k))
            .cloned();
        match idle {
            Some(idle) => {
                table.paused.remove(&idle);
            }
            // All running: a pause for a file nobody reads yet is not kept.
            None if !table.slots.contains_key(key) => return,
            None => {}
        }
    }
    table.paused.insert(key.clone());
}

/// Requests queued for the peer of connection `conn`, oldest first, to send it as
/// `PauseTransfer` in this order.
pub fn take_peer_requests(conn: usize) -> Vec<PeerRequest> {
    let mut table = TABLE.lock().unwrap();
    let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut table.requests)
        .into_iter()
        .partition(|(c, _)| *c == conn);
    table.requests = rest;
    mine.into_iter().map(|(_, r)| r).collect()
}

/// Changes whenever a request for a peer is queued: wait on it, then
/// [`take_peer_requests`].
pub fn peer_requests_queued() -> tokio::sync::watch::Receiver<()> {
    REQUESTS_QUEUED.subscribe()
}

/// Keeps connection `conn`'s cancels, pauses and peer requests remembered; dropping the
/// last guard for it forgets them. Hold one for as long as the connection is used.
#[must_use]
pub struct ConnectionGuard(usize);

impl ConnectionGuard {
    pub fn new(conn: usize) -> Self {
        *TABLE.lock().unwrap().live.entry(conn).or_default() += 1;
        Self(conn)
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        let conn = self.0;
        let mut table = TABLE.lock().unwrap();
        if let Some(n) = table.live.get_mut(&conn) {
            *n -= 1;
            if *n > 0 {
                return;
            }
        }
        table.live.remove(&conn);
        table.cancelled.retain(|k| k.conn != conn);
        table.paused.retain(|k| k.conn != conn);
        table.requests.retain(|(c, _)| *c != conn);
        table.lingering.retain(|k, _| k.conn != conn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_of_one_file_share_a_transfer() {
        // A connection id no other test uses: the table is process-wide.
        let conn = usize::MAX - 1;
        let live = ConnectionGuard::new(conn);
        let path = vec!["dir".to_string(), "a.bin".to_string()];
        let a = track(conn, TransferDirection::Incoming, 7, &path, 100);
        let b = track(conn, TransferDirection::Incoming, 7, &path, 120);
        let other = track(conn, TransferDirection::Outgoing, 7, &path, 100);
        let idle = track(conn, TransferDirection::Incoming, 9, &path, 100);
        assert!(list(conn).is_empty(), "young transfers are not listed");
        a.advance(40, 40);
        b.advance(10, 10);
        other.advance(1, 1);
        std::thread::sleep(MIN_LISTED_AGE);
        let listed = list(conn);
        assert_eq!(
            listed.len(),
            2,
            "a file held open without reading is not listed"
        );
        assert_eq!(listed[0].name, "a.bin");
        assert_eq!((listed[0].position, listed[0].moved), (40, 50));
        assert_eq!(
            listed[0].size, 120,
            "a reader that joins sees the file grown"
        );
        drop(a);
        assert_eq!(list(conn).len(), 2, "one reader left");
        assert!(cancel(listed[0].id));
        assert!(cancel(listed[0].id), "cancelling twice is harmless");
        assert!(b.is_cancelled() && !other.is_cancelled());
        drop(b);
        drop(other);
        drop(idle);
        assert!(list(conn).is_empty());
        assert!(!cancel(listed[0].id));
        // Reading the cancelled file again starts out cancelled; other files do not.
        let again = track(conn, TransferDirection::Incoming, 7, &path, 100);
        assert!(again.is_cancelled());
        let other_offer = track(conn, TransferDirection::Incoming, 8, &path, 100);
        assert!(!other_offer.is_cancelled());
        drop(again);
        let remembered = |conn| {
            TABLE
                .lock()
                .unwrap()
                .cancelled
                .iter()
                .filter(|k| k.conn == conn)
                .count()
        };
        assert_eq!(
            remembered(conn),
            1,
            "a file cancelled twice is remembered once"
        );
        // Once the connection is gone, nothing is remembered for it, even for a
        // transfer that ends later; a later connection with the same id starts clean.
        drop(live);
        assert_eq!(remembered(conn), 0);
        assert!(cancel(other_offer.entry.id));
        drop(other_offer);
        assert_eq!(remembered(conn), 0);
        let _live = ConnectionGuard::new(conn);
        assert!(!track(conn, TransferDirection::Incoming, 7, &path, 100).is_cancelled());
    }

    #[test]
    fn outgoing_pauses_are_requested_from_the_peer_and_released_when_done() {
        let conn = usize::MAX - 4;
        let live = ConnectionGuard::new(conn);
        let path = vec!["small.bin".to_string()];
        let req = |paused| PeerRequest {
            offer_id: 5,
            path: path.clone(),
            paused,
        };
        let guard = track(conn, TransferDirection::Outgoing, 5, &path, 10);
        assert!(set_paused(guard.entry.id, true));
        assert_eq!(take_peer_requests(conn), vec![req(true)]);
        // It ends here while paused (all sent): the peer is asked to resume.
        drop(guard);
        assert_eq!(take_peer_requests(conn), vec![req(false)]);
        assert!(take_peer_requests(conn).is_empty());
        // Cancelling a paused one resumes the peer, once.
        let again = track(conn, TransferDirection::Outgoing, 5, &path, 10);
        assert!(!again.is_paused());
        assert!(set_paused(again.entry.id, true));
        assert!(cancel(again.entry.id));
        drop(again);
        assert_eq!(take_peer_requests(conn), vec![req(true), req(false)]);
        // A cancelled one cannot be paused again.
        let cancelled = track(conn, TransferDirection::Outgoing, 7, &path, 10);
        assert!(cancel(cancelled.entry.id));
        assert!(!set_paused(cancelled.entry.id, true));
        assert!(!cancelled.is_paused());
        drop(cancelled);
        assert!(take_peer_requests(conn).is_empty());
        // A stream joining a paused transfer asks the peer to pause again.
        let first = track(conn, TransferDirection::Outgoing, 8, &path, 10);
        assert!(set_paused(first.entry.id, true));
        let req8 = |paused| PeerRequest {
            offer_id: 8,
            path: path.clone(),
            paused,
        };
        assert_eq!(take_peer_requests(conn), vec![req8(true)]);
        let second = track(conn, TransferDirection::Outgoing, 8, &path, 10);
        assert_eq!(take_peer_requests(conn), vec![req8(true)]);
        drop(first);
        drop(second);
        assert_eq!(take_peer_requests(conn), vec![req8(false)]);
        // Pausing one that ended does nothing, and asks nothing of the peer.
        assert!(!set_paused(u64::MAX, true));
        assert!(take_peer_requests(conn).is_empty());
        // A connection that is gone gets no requests.
        let late = track(conn, TransferDirection::Outgoing, 6, &path, 10);
        assert!(set_paused(late.entry.id, true));
        drop(live);
        assert!(take_peer_requests(conn).is_empty());
        drop(late);
        assert!(take_peer_requests(conn).is_empty());
    }

    #[tokio::test]
    async fn a_reopened_stream_continues_a_paused_outgoing_transfer() {
        let conn = usize::MAX - 6;
        let _live = ConnectionGuard::new(conn);
        let path = vec!["seek.bin".to_string()];
        let req = |paused| PeerRequest {
            offer_id: 9,
            path: path.clone(),
            paused,
        };
        let first = track(conn, TransferDirection::Outgoing, 9, &path, 10);
        assert!(set_paused(first.entry.id, true));
        assert_eq!(take_peer_requests(conn), vec![req(true)]);
        // The peer's reader seeks: its old stream ends, then a new one arrives.
        drop(first);
        assert!(take_peer_requests(conn).is_empty(), "not resumed at once");
        let second = track(conn, TransferDirection::Outgoing, 9, &path, 10);
        assert!(second.is_paused(), "the new stream continues the pause");
        assert_eq!(take_peer_requests(conn), vec![req(true)]);
        // Once it ends for good, the peer is resumed after the grace period.
        drop(second);
        tokio::time::sleep(RELEASE_GRACE + Duration::from_millis(300)).await;
        assert_eq!(take_peer_requests(conn), vec![req(false)]);
        assert!(!track(conn, TransferDirection::Outgoing, 9, &path, 10).is_paused());
    }

    #[tokio::test]
    async fn paused_transfers_hold_readers_until_resumed_or_cancelled() {
        let conn = usize::MAX - 3;
        let _live = ConnectionGuard::new(conn);
        let path = vec!["p.bin".to_string()];
        let first = track(conn, TransferDirection::Incoming, 3, &path, 10);
        // Not paused: no wait.
        tokio::time::timeout(Duration::from_millis(100), first.unpaused())
            .await
            .unwrap();
        assert!(set_paused(first.entry.id, true));
        assert!(first.is_paused());
        assert!(
            tokio::time::timeout(Duration::from_millis(100), first.unpaused())
                .await
                .is_err()
        );
        // A reader joining the running transfer is paused too.
        let guard = Arc::new(track(conn, TransferDirection::Incoming, 3, &path, 10));
        assert!(guard.is_paused());
        let wait = {
            let guard = Arc::clone(&guard);
            tokio::spawn(async move { guard.unpaused().await })
        };
        tokio::task::yield_now().await;
        assert!(!wait.is_finished());
        // The peer's request, by file, resumes it.
        let file = TransferFile {
            direction: TransferDirection::Incoming,
            offer_id: 3,
            path: path.clone(),
        };
        set_paused_file(conn, &file, false);
        tokio::time::timeout(Duration::from_secs(1), wait)
            .await
            .unwrap()
            .unwrap();
        assert!(!guard.is_paused());

        // Cancelling releases paused readers too, and it is no longer shown as paused.
        set_paused_file(conn, &file, true);
        let wait = {
            let guard = Arc::clone(&guard);
            tokio::spawn(async move { guard.unpaused().await })
        };
        assert!(cancel(guard.entry.id));
        tokio::time::timeout(Duration::from_secs(1), wait)
            .await
            .unwrap()
            .unwrap();
        guard.advance(1, 1);
        let mut shown = Vec::new();
        while shown.is_empty() {
            tokio::time::sleep(Duration::from_millis(100)).await;
            shown = list(conn);
        }
        assert!(shown[0].cancelled && !shown[0].paused);
    }

    #[test]
    fn incoming_pauses_end_with_the_transfer() {
        let conn = usize::MAX - 5;
        let _live = ConnectionGuard::new(conn);
        let path = vec!["q.bin".to_string()];
        let file = TransferFile {
            direction: TransferDirection::Incoming,
            offer_id: 4,
            path: path.clone(),
        };
        // The peer pauses it before anything reads it: the first reader is held.
        set_paused_file(conn, &file, true);
        let guard = track(conn, TransferDirection::Incoming, 4, &path, 10);
        assert!(guard.is_paused());
        // Once its last reader is done, the pause is forgotten: nothing would be listed
        // to resume it from.
        drop(guard);
        assert!(!track(conn, TransferDirection::Incoming, 4, &path, 10).is_paused());
        // Also when it was cancelled while paused.
        set_paused_file(conn, &file, true);
        let guard = track(conn, TransferDirection::Incoming, 4, &path, 10);
        assert!(cancel(guard.entry.id));
        drop(guard);
        assert!(!TABLE.lock().unwrap().paused.iter().any(|k| k.conn == conn));
    }

    #[test]
    fn remembered_pauses_stay_capped() {
        let conn = usize::MAX - 7;
        let _live = ConnectionGuard::new(conn);
        let file = |i: usize| TransferFile {
            direction: TransferDirection::Incoming,
            offer_id: 1,
            path: vec![format!("f{i}")],
        };
        // Every remembered pause belongs to a running transfer: none can be forgotten.
        let running: Vec<_> = (0..REMEMBERED_PAUSES)
            .map(|i| {
                let guard = track(conn, TransferDirection::Incoming, 1, &file(i).path, 1);
                set_paused_file(conn, &file(i), true);
                guard
            })
            .collect();
        let count = || {
            TABLE
                .lock()
                .unwrap()
                .paused
                .iter()
                .filter(|k| k.conn == conn)
                .count()
        };
        assert_eq!(count(), REMEMBERED_PAUSES);
        for i in REMEMBERED_PAUSES..REMEMBERED_PAUSES + 10 {
            set_paused_file(conn, &file(i), true);
        }
        assert_eq!(
            count(),
            REMEMBERED_PAUSES,
            "pauses for unopened files are not kept"
        );
        drop(running);
    }

    #[tokio::test]
    async fn cancelled_wakes_waiters() {
        let conn = usize::MAX - 2;
        let guard = track(conn, TransferDirection::Outgoing, 1, &["f".into()], 1);
        let wait = tokio::spawn(guard.cancelled());
        tokio::task::yield_now().await;
        assert!(!wait.is_finished());
        assert!(cancel(guard.entry.id));
        let cancelled = tokio::time::timeout(Duration::from_secs(1), wait).await;
        assert!(cancelled.unwrap().unwrap());

        let other = track(conn, TransferDirection::Outgoing, 2, &["g".into()], 1);
        let wait = tokio::spawn(other.cancelled());
        drop(other);
        let cancelled = tokio::time::timeout(Duration::from_secs(1), wait).await;
        assert!(
            !cancelled.unwrap().unwrap(),
            "an ended transfer is not cancelled"
        );
    }
}
