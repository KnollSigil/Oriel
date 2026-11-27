//! Process-wide table of file transfers in progress, for showing them to the user.
//!
//! A transfer is one file of an offer moving over one connection, in one direction: read
//! through this process's transfer mount ([`TransferDirection::Incoming`]) or served to the
//! peer ([`TransferDirection::Outgoing`]). Every reader of the same file (a copy plus a
//! thumbnailer, a reopened stream after a seek) adds to the same entry, which ends when the
//! last of them does. A transfer can be cancelled: its reads then fail, and so do later
//! reads of the same file of the same offer (readers reopen streams that fail).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

/// Transfers younger than this are not listed: file managers and thumbnailers open files
/// briefly to sniff their type or render a preview.
const MIN_LISTED_AGE: Duration = Duration::from_secs(1);
/// How many cancelled files are remembered, so reopening them fails too.
const REMEMBERED_CANCELS: usize = 256;

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
    pub cancelled: bool,
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
    size: u64,
    started: Instant,
    position: AtomicU64,
    moved: AtomicU64,
    cancel: tokio::sync::watch::Sender<bool>,
}

struct Slot {
    entry: Arc<Entry>,
    readers: usize,
}

struct Table {
    slots: HashMap<Key, Slot>,
    cancelled: VecDeque<Key>,
    next_id: u64,
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(|| {
    Mutex::new(Table {
        slots: HashMap::new(),
        cancelled: VecDeque::new(),
        next_id: 1,
    })
});

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
    let slot = table.slots.entry(key.clone()).or_insert_with(|| Slot {
        entry: Arc::new(Entry {
            id: next_id,
            size,
            started: Instant::now(),
            position: AtomicU64::new(0),
            moved: AtomicU64::new(0),
            cancel: tokio::sync::watch::channel(was_cancelled).0,
        }),
        readers: 0,
    });
    slot.readers += 1;
    let entry = Arc::clone(&slot.entry);
    if entry.id == next_id {
        table.next_id += 1;
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
        if let Some(slot) = table.slots.get_mut(&self.key) {
            slot.readers -= 1;
            if slot.readers == 0 {
                table.slots.remove(&self.key);
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
        .filter(|(key, slot)| key.conn == conn && slot.entry.started.elapsed() >= MIN_LISTED_AGE)
        .map(|(key, slot)| {
            let e = &slot.entry;
            (
                e.started,
                TransferInfo {
                    id: e.id,
                    direction: key.direction,
                    name: key.path.last().cloned().unwrap_or_default(),
                    size: e.size,
                    position: e.position.load(Ordering::Relaxed).min(e.size),
                    moved: e.moved.load(Ordering::Relaxed),
                    cancelled: *e.cancel.borrow(),
                },
            )
        })
        .collect();
    out.sort_by_key(|(started, info)| (*started, info.id));
    out.into_iter().map(|(_, info)| info).collect()
}

/// Cancel transfer `id`: its readers' pending and later reads fail, and so does reading
/// that file of that offer again. `false` if it ended.
pub fn cancel(id: u64) -> bool {
    let mut table = TABLE.lock().unwrap();
    let Some((key, slot)) = table.slots.iter().find(|(_, slot)| slot.entry.id == id) else {
        return false;
    };
    slot.entry.cancel.send_replace(true);
    let key = key.clone();
    if table.cancelled.len() == REMEMBERED_CANCELS {
        table.cancelled.pop_front();
    }
    table.cancelled.push_back(key);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_of_one_file_share_a_transfer() {
        // A connection id no other test uses: the table is process-wide.
        let conn = usize::MAX - 1;
        let path = vec!["dir".to_string(), "a.bin".to_string()];
        let a = track(conn, TransferDirection::Incoming, 7, &path, 100);
        let b = track(conn, TransferDirection::Incoming, 7, &path, 100);
        let other = track(conn, TransferDirection::Outgoing, 7, &path, 100);
        assert!(list(conn).is_empty(), "young transfers are not listed");
        a.advance(40, 40);
        b.advance(10, 10);
        std::thread::sleep(MIN_LISTED_AGE);
        let listed = list(conn);
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].name, "a.bin");
        assert_eq!((listed[0].position, listed[0].moved), (40, 50));
        drop(a);
        assert_eq!(list(conn).len(), 2, "one reader left");
        assert!(cancel(listed[0].id));
        assert!(b.is_cancelled() && !other.is_cancelled());
        drop(b);
        drop(other);
        assert!(list(conn).is_empty());
        assert!(!cancel(listed[0].id));
        // Reading the cancelled file again starts out cancelled; other files do not.
        let again = track(conn, TransferDirection::Incoming, 7, &path, 100);
        assert!(again.is_cancelled());
        let other_offer = track(conn, TransferDirection::Incoming, 8, &path, 100);
        assert!(!other_offer.is_cancelled());
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
