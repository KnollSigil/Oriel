import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Templates as T

ApplicationWindow {
    id: root
    visible: true
    width: 960
    height: 640
    minimumWidth: 640
    minimumHeight: 420
    title: backend.activeLabel.length > 0
        ? "RemoteWayland — " + backend.activeLabel
        : "RemoteWayland"

    required property var backend
    required property var hostsModel
    required property var appsModel
    required property var programsModel
    required property var micsModel
    required property var terminalsModel

    Component.onCompleted: {
        if (serverList.count > 0) {
            serverList.currentIndex = 0
            serverList.forceActiveFocus()
        } else {
            manageBtn.forceActiveFocus()
        }
    }

    // Reusable unified split-button for default + selectable terminal emulator
    component TerminalSplitButton: Rectangle {
        id: splitRoot
        property string primaryText: "New Terminal"
        property string primaryToolTip: "Open remote terminal"
        property string dropToolTip: "Choose terminal emulator"
        property bool tabNavigable: false
        property int dropWidth: 28
        property alias primaryButton: splitPrimaryBtn
        property alias dropButton: splitDropBtn
        signal triggered(string binary)

        radius: 4
        color: root.palette.button
        border.width: 1
        border.color: (splitPrimaryBtn.hovered || splitDropBtn.hovered
            || splitPrimaryBtn.activeFocus || splitDropBtn.activeFocus
            || splitMenu.visible)
            ? root.palette.highlight
            : root.palette.mid
        implicitHeight: splitRow.implicitHeight
        implicitWidth: splitRow.implicitWidth

        RowLayout {
            id: splitRow
            anchors.fill: parent
            spacing: 0

            ToolButton {
                id: splitPrimaryBtn
                Layout.fillHeight: true
                activeFocusOnTab: splitRoot.tabNavigable
                text: splitRoot.primaryText
                icon.name: "utilities-terminal"
                display: AbstractButton.TextBesideIcon
                onClicked: splitRoot.triggered("")
                // Dedicated tooltips: instances of this component also live in runningPopup's
                // window (see note on the shared ToolTip instance).
                ToolTip {
                    popupType: Popup.Window
                    visible: splitPrimaryBtn.hovered
                    delay: 400
                    text: splitRoot.primaryToolTip
                }
            }

            Rectangle {
                Layout.fillHeight: true
                Layout.topMargin: 4
                Layout.bottomMargin: 4
                implicitWidth: 1
                color: root.palette.mid
                opacity: 0.7
            }

            ToolButton {
                id: splitDropBtn
                Layout.fillHeight: true
                implicitWidth: splitRoot.dropWidth
                activeFocusOnTab: splitRoot.tabNavigable
                icon.name: "arrow-down"
                display: AbstractButton.IconOnly
                Accessible.name: splitRoot.dropToolTip
                onClicked: splitMenu.popup(splitRoot, 0, splitRoot.height + 2)
                ToolTip {
                    popupType: Popup.Window
                    visible: splitDropBtn.hovered
                    delay: 400
                    text: splitRoot.dropToolTip
                }
            }
        }

        Menu {
            id: splitMenu
            // Must be its own window: it opens from inside runningPopup's window too.
            popupType: Popup.Window
            Instantiator {
                model: root.terminalsModel
                delegate: MenuItem {
                    required property string label
                    required property string binary
                    text: label
                    onTriggered: splitRoot.triggered(binary)
                }
                onObjectAdded: (index, object) => splitMenu.insertItem(index, object)
                onObjectRemoved: (index, object) => splitMenu.removeItem(object)
            }
        }
    }

    // Standard ComboBox whose dropdown is a native popup window.
    //
    // `popup.popupType: Popup.Window` on a plain ComboBox doesn't work on Wayland:
    // QQuickComboBox::focusOutEvent closes a window-type popup on *any* focus-out, and the
    // dropdown's xdg_popup grab takes keyboard focus from the ComboBox's window, so the dropdown
    // closes about one frame after it opens. Instead, the ComboBox's own `popup` is an invisible
    // in-window stub (which it may close whenever it likes); opening the stub (click, Space,
    // Alt+Down, ...) hands off to a native-window Menu of plain MenuItems — the same thing the
    // KDE desktop style shows as its dropdown.
    //
    // The Menu builds its own MenuItems rather than reusing `delegateModel`: the style's default
    // popup is still instantiated despite the override and already consumes `delegateModel`.
    //
    // Difference from ComboBox: picking an item emits `activated(index)` but does NOT set
    // `currentIndex` (doing that from QML would break the owner's `currentIndex` binding).
    // Owners update `currentIndex` themselves, usually via the backend binding.
    component WindowComboBox: ComboBox {
        id: combo
        property alias dropdown: dropdownMenu

        down: pressed || dropdownMenu.visible

        popup: T.Popup {
            implicitWidth: 0
            implicitHeight: 0
            padding: 0
            closePolicy: T.Popup.NoAutoClose
            exit: Transition {} // styles read popup.exit.running
            onOpened: {
                close()
                dropdownMenu.currentIndex = combo.currentIndex
                dropdownMenu.popup(combo, 0, combo.height)
            }
        }

        Menu {
            id: dropdownMenu
            popupType: Popup.Window

            Instantiator {
                model: combo.model
                delegate: MenuItem {
                    required property int index
                    required property var model
                    text: combo.textRole.length > 0 ? model[combo.textRole] : model.modelData
                    onTriggered: combo.activated(index)
                }
                onObjectAdded: (index, object) => dropdownMenu.insertItem(index, object)
                onObjectRemoved: (index, object) => dropdownMenu.removeItem(object)
            }
        }
    }

    // Helper functions for clean keyboard navigation
    function focusSearch() {
        searchField.forceActiveFocus()
        searchField.cursorPosition = searchField.text.length
    }

    function focusServerList() {
        if (serverList.count > 0) {
            if (serverList.currentIndex < 0 || serverList.currentIndex >= serverList.count) {
                serverList.currentIndex = 0
            }
            serverList.forceActiveFocus()
        } else {
            manageBtn.forceActiveFocus()
        }
    }

    function focusAppGrid() {
        if (appGrid.visible && appGrid.count > 0) {
            if (appGrid.currentIndex < 0 || appGrid.currentIndex >= appGrid.count) {
                appGrid.currentIndex = 0
            }
            appGrid.forceActiveFocus()
        } else {
            focusSearch()
        }
    }

    function toggleRunningApps() {
        if (runningPopup.visible) {
            runningPopup.close()
        } else if (backend.programCount > 0) {
            settingsPopup.close()
            runningPopup.open()
        }
    }

    function toggleSettingsPopup() {
        if (settingsPopup.visible) {
            settingsPopup.close()
        } else {
            runningPopup.close()
            backend.refresh_mics()
            settingsPopup.open()
        }
    }

    // Global keyboard shortcuts (disabled while modal Manage Connections dialog is open)
    Shortcut {
        enabled: !manageDialog.opened
        sequences: [StandardKey.Find, "Ctrl+L"]
        onActivated: {
            searchField.forceActiveFocus()
            searchField.selectAll()
        }
    }
    Shortcut {
        enabled: !manageDialog.opened
        sequence: "Ctrl+1"
        onActivated: root.focusServerList()
    }
    Shortcut {
        enabled: !manageDialog.opened
        sequence: "Ctrl+2"
        onActivated: root.focusAppGrid()
    }
    Shortcut {
        enabled: !manageDialog.opened
        sequence: "Ctrl+3"
        onActivated: root.toggleRunningApps()
    }
    Shortcut {
        enabled: !manageDialog.opened
        sequence: "Ctrl+T"
        onActivated: {
            if (backend.isConnected) {
                backend.open_terminal("")
            }
        }
    }
    Shortcut {
        enabled: !manageDialog.opened
        sequence: "Ctrl+,"
        onActivated: root.toggleSettingsPopup()
    }

    // -----------------------------------------------------------------------
    // Bottom status bar (ApplicationWindow footer)
    // -----------------------------------------------------------------------
    footer: ToolBar {
        position: ToolBar.Footer

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 10
            anchors.rightMargin: 10
            spacing: 10

            Label {
                text: backend.activeLabel.length > 0
                    ? backend.activeLabel + " — " + backend.statusText
                    : backend.statusText
                elide: Text.ElideRight
                Layout.fillWidth: true
            }

            Label {
                visible: backend.rttText.length > 0
                text: backend.isCongested ? backend.rttText + " (congested)" : backend.rttText
                color: backend.isCongested ? "#e6a23c" : root.palette.windowText
                opacity: 0.85
            }

            Label {
                text: "Mic: " + backend.micSummary
                opacity: 0.75
                elide: Text.ElideRight
                Layout.maximumWidth: 200
            }

            Button {
                id: connectBtn
                visible: backend.activeTarget.length > 0
                activeFocusOnTab: true
                text: backend.isConnected || backend.isConnecting ? "Disconnect" : "Connect"
                KeyNavigation.tab: serverList
                KeyNavigation.backtab: settingsBtn
                onClicked: {
                    if (backend.isConnected || backend.isConnecting) {
                        backend.disconnect_host(backend.activeTarget)
                    } else {
                        backend.connect_host(backend.activeTarget)
                        root.focusSearch()
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Main layout: SplitView (left sidebar + right launcher)
    // -----------------------------------------------------------------------
    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // All attached `ToolTip.*` share one per-engine ToolTip instance; make it a native
        // (non-grabbing, non-focus-taking) popup so tooltips aren't clipped at the window edge.
        // Needs an Item to attach to.
        //
        // NOTE: Qt creates a popup's native window once, parented to whatever window it is first
        // shown over, and never re-parents it. So the shared instance must only be used by items
        // in the main window. Items inside popup windows (settingsPopup, runningPopup, Menus)
        // need a dedicated child `ToolTip { popupType: Popup.Window; visible: <button>.hovered }`,
        // or their tooltip is positioned relative to the wrong window.
        Component.onCompleted: ToolTip.toolTip.popupType = Popup.Window

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            // --- Left Server Sidebar ---
            Pane {
                SplitView.preferredWidth: 210
                SplitView.minimumWidth: 150
                SplitView.maximumWidth: 360
                padding: 8

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 4

                        Label {
                            text: "Servers"
                            font.bold: true
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }

                        ToolButton {
                            id: manageBtn
                            icon.name: "configure"
                            display: AbstractButton.IconOnly
                            Accessible.name: "Manage connections"
                            activeFocusOnTab: true
                            ToolTip.visible: hovered
                            ToolTip.text: "Manage connections"
                            onClicked: manageDialog.open()
                            KeyNavigation.down: serverList
                            KeyNavigation.right: searchField
                        }
                    }

                    Frame {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        padding: 2

                        ListView {
                            id: serverList
                            anchors.fill: parent
                            clip: true
                            spacing: 2
                            model: root.hostsModel
                            activeFocusOnTab: true
                            keyNavigationEnabled: true
                            keyNavigationWraps: false
                            highlightFollowsCurrentItem: true

                            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                            KeyNavigation.right: searchField
                            KeyNavigation.tab: searchField
                            KeyNavigation.backtab: manageBtn

                            Keys.onReturnPressed: {
                                if (currentIndex >= 0) {
                                    backend.select_host_at(currentIndex)
                                    root.focusSearch()
                                }
                            }
                            Keys.onEnterPressed: {
                                if (currentIndex >= 0) {
                                    backend.select_host_at(currentIndex)
                                    root.focusSearch()
                                }
                            }
                            Keys.onSpacePressed: {
                                if (currentIndex >= 0) {
                                    backend.select_host_at(currentIndex)
                                    root.focusSearch()
                                }
                            }

                            delegate: ItemDelegate {
                                id: hostDelegate
                                required property int index
                                required property string label
                                required property string ssh_target
                                required property string status
                                required property bool is_active
                                required property string rtt_text
                                required property bool congested
                                required property int program_count

                                width: serverList.width
                                highlighted: (serverList.activeFocus && serverList.currentIndex === index) || is_active
                                onClicked: {
                                    serverList.currentIndex = index
                                    backend.select_host(ssh_target)
                                    root.focusSearch()
                                }

                                contentItem: RowLayout {
                                    spacing: 8

                                    Rectangle {
                                        width: 8
                                        height: 8
                                        radius: 4
                                        color: status === "connected"
                                            ? (congested ? "#e6a23c" : "#2ecc71")
                                            : (status === "connecting" ? "#f1c40f" : root.palette.mid)
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 1

                                        Label {
                                            text: hostDelegate.label
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                            font.bold: hostDelegate.is_active
                                            color: hostDelegate.highlighted
                                                ? root.palette.highlightedText
                                                : root.palette.windowText
                                        }

                                        Label {
                                            text: hostDelegate.rtt_text.length > 0
                                                ? hostDelegate.rtt_text
                                                : hostDelegate.ssh_target
                                            font.pointSize: Math.max(8, root.font.pointSize - 2)
                                            opacity: 0.7
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                            color: hostDelegate.highlighted
                                                ? root.palette.highlightedText
                                                : root.palette.windowText
                                        }
                                    }

                                    Label {
                                        visible: hostDelegate.program_count > 0
                                        text: hostDelegate.program_count.toString()
                                        font.pointSize: Math.max(8, root.font.pointSize - 2)
                                        opacity: 0.8
                                        color: hostDelegate.highlighted
                                            ? root.palette.highlightedText
                                            : root.palette.windowText
                                    }
                                }
                            }

                            Label {
                                anchors.centerIn: parent
                                visible: serverList.count === 0
                                text: "No servers yet.\nClick ⚙ to add one."
                                horizontalAlignment: Text.AlignHCenter
                                opacity: 0.6
                                wrapMode: Text.WordWrap
                                width: parent.width - 16
                            }
                        }
                    }
                }
            }

            // --- Right Main Pane ---
            Pane {
                SplitView.fillWidth: true
                padding: 10

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    // Top bar: Search + Split New Terminal + Settings
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        TextField {
                            id: searchField
                            Layout.fillWidth: true
                            placeholderText: "Type to search apps, or >command to run directly…"
                            text: backend.searchQuery
                            selectByMouse: true
                            activeFocusOnTab: true

                            onTextEdited: backend.set_search_query(text)

                            KeyNavigation.tab: termSplit.enabled ? termSplit.primaryButton : settingsBtn
                            KeyNavigation.backtab: serverList

                            Keys.onReturnPressed: {
                                if (backend.isConnected) {
                                    backend.launch_primary_action()
                                } else if (backend.activeTarget.length > 0 && !backend.isConnecting) {
                                    backend.connect_host(backend.activeTarget)
                                }
                            }
                            Keys.onEnterPressed: {
                                if (backend.isConnected) {
                                    backend.launch_primary_action()
                                } else if (backend.activeTarget.length > 0 && !backend.isConnecting) {
                                    backend.connect_host(backend.activeTarget)
                                }
                            }
                            Keys.onEscapePressed: {
                                if (text.length > 0) {
                                    backend.set_search_query("")
                                } else {
                                    root.focusServerList()
                                }
                            }
                            Keys.onLeftPressed: (event) => {
                                if (cursorPosition === 0 && selectedText.length === 0) {
                                    root.focusServerList()
                                    event.accepted = true
                                } else {
                                    event.accepted = false
                                }
                            }
                            Keys.onRightPressed: (event) => {
                                if (cursorPosition === text.length && selectedText.length === 0) {
                                    if (termSplit.enabled) {
                                        termSplit.primaryButton.forceActiveFocus()
                                    } else {
                                        settingsBtn.forceActiveFocus()
                                    }
                                    event.accepted = true
                                } else {
                                    event.accepted = false
                                }
                            }
                            Keys.onDownPressed: {
                                if (appGrid.visible && appGrid.count > 0) {
                                    appGrid.currentIndex = 0
                                    appGrid.forceActiveFocus()
                                }
                            }
                        }

                        // Unified Split "New Terminal" button
                        TerminalSplitButton {
                            id: termSplit
                            enabled: backend.isConnected
                            opacity: enabled ? 1.0 : 0.45
                            implicitHeight: Math.max(searchField.implicitHeight, primaryButton.implicitHeight)
                            primaryText: "New Terminal"
                            primaryToolTip: "Open remote terminal (Ctrl+T)"
                            dropToolTip: "Choose terminal emulator for remote PTY"
                            tabNavigable: true
                            onTriggered: (binary) => backend.open_terminal(binary)

                            primaryButton.KeyNavigation.left: searchField
                            primaryButton.KeyNavigation.right: dropButton
                            primaryButton.KeyNavigation.tab: dropButton
                            primaryButton.KeyNavigation.backtab: searchField

                            dropButton.KeyNavigation.left: primaryButton
                            dropButton.KeyNavigation.right: runningAppsBtn.enabled ? runningAppsBtn : settingsBtn
                            dropButton.KeyNavigation.tab: runningAppsBtn.enabled ? runningAppsBtn : settingsBtn
                            dropButton.KeyNavigation.backtab: primaryButton
                        }

                        // Running Applications popup button (hidden by default until clicked or Ctrl+3)
                        Button {
                            id: runningAppsBtn
                            activeFocusOnTab: true
                            visible: backend.isConnected
                            enabled: backend.programCount > 0
                            icon.name: "system-run"
                            text: backend.programCount > 0
                                ? "Running (" + backend.programCount + ")"
                                : "Running"
                            ToolTip.visible: hovered
                            ToolTip.delay: 400
                            ToolTip.text: "Show running applications and terminals (Ctrl+3)"
                            onClicked: root.toggleRunningApps()
                            KeyNavigation.left: termSplit.enabled ? termSplit.dropButton : searchField
                            KeyNavigation.right: settingsBtn
                            KeyNavigation.tab: settingsBtn
                            KeyNavigation.backtab: termSplit.enabled ? termSplit.dropButton : searchField

                            Popup {
                                id: runningPopup
                                // implicitWidth, not width: window-type popups resize to their
                                // implicit size when opened, which overrode an explicit width.
                                implicitWidth: Math.min(480, root.width - 32)
                                // Native popup window (xdg_popup on Wayland) so it can extend
                                // past the main window. Margins only matter if the platform
                                // falls back to an in-window Popup.Item.
                                popupType: Popup.Window
                                x: parent.width - width
                                y: parent.height + 4
                                margins: 16
                                padding: 12
                                modal: false
                                focus: true
                                closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent
                                onOpened: {
                                    Qt.callLater(function() {
                                        if (runningAppsList.count > 0) {
                                            if (runningAppsList.currentIndex < 0 || runningAppsList.currentIndex >= runningAppsList.count) {
                                                runningAppsList.currentIndex = 0
                                            }
                                            runningAppsList.forceActiveFocus()
                                        }
                                    })
                                }
                                onClosed: {
                                    if (runningAppsBtn.visible && runningAppsBtn.enabled) {
                                        runningAppsBtn.forceActiveFocus()
                                    } else {
                                        root.focusSearch()
                                    }
                                }

                                ColumnLayout {
                                    width: parent.width
                                    spacing: 8

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 8

                                        Label {
                                            text: "Running Applications (" + backend.programCount + ")"
                                            font.bold: true
                                            Layout.fillWidth: true
                                        }

                                        Label {
                                            text: "Del: Stop · Enter: Attach"
                                            font.pointSize: Math.max(8, root.font.pointSize - 2)
                                            opacity: 0.65
                                        }
                                    }

                                    ListView {
                                        id: runningAppsList
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: Math.min(Math.max(contentHeight, 44), 320)
                                        clip: true
                                        spacing: 4
                                        model: root.programsModel
                                        keyNavigationEnabled: true
                                        highlightFollowsCurrentItem: true

                                        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                                        Keys.onUpPressed: (event) => {
                                            if (currentIndex > 0) {
                                                decrementCurrentIndex()
                                            }
                                            event.accepted = true
                                        }
                                        Keys.onDownPressed: (event) => {
                                            if (currentIndex + 1 < count) {
                                                incrementCurrentIndex()
                                            }
                                            event.accepted = true
                                        }
                                        Keys.onDeletePressed: (event) => {
                                            if (currentIndex >= 0) {
                                                backend.stop_program_at(currentIndex)
                                            }
                                            event.accepted = true
                                        }
                                        Keys.onPressed: (event) => {
                                            if (event.key === Qt.Key_Backspace && currentIndex >= 0) {
                                                backend.stop_program_at(currentIndex)
                                                event.accepted = true
                                            }
                                        }
                                        Keys.onReturnPressed: (event) => {
                                            if (currentIndex >= 0) {
                                                backend.reattach_pty_at(currentIndex)
                                            }
                                            event.accepted = true
                                        }
                                        Keys.onEnterPressed: (event) => {
                                            if (currentIndex >= 0) {
                                                backend.reattach_pty_at(currentIndex)
                                            }
                                            event.accepted = true
                                        }
                                        Keys.onEscapePressed: (event) => {
                                            runningPopup.close()
                                            event.accepted = true
                                        }

                                        delegate: ItemDelegate {
                                            id: progRow
                                            required property int index
                                            required property string launch_id
                                            required property string name
                                            required property string state
                                            required property bool can_reattach
                                            required property bool term_sent

                                            width: runningAppsList.width
                                            padding: 6
                                            highlighted: runningAppsList.currentIndex === index
                                            onClicked: {
                                                runningAppsList.currentIndex = index
                                                runningAppsList.forceActiveFocus()
                                            }

                                            contentItem: RowLayout {
                                                spacing: 8

                                                Rectangle {
                                                    Layout.alignment: Qt.AlignVCenter
                                                    width: 8
                                                    height: 8
                                                    radius: 4
                                                    color: progRow.term_sent
                                                        ? "#e6a23c"
                                                        : (progRow.state === "Running" ? "#2ecc71" : "#f1c40f")
                                                }

                                                ColumnLayout {
                                                    Layout.fillWidth: true
                                                    Layout.alignment: Qt.AlignVCenter
                                                    spacing: 0

                                                    Label {
                                                        text: progRow.name
                                                        Layout.fillWidth: true
                                                        elide: Text.ElideRight
                                                        font.bold: progRow.highlighted
                                                        color: progRow.highlighted
                                                            ? root.palette.highlightedText
                                                            : root.palette.windowText
                                                    }

                                                    Label {
                                                        text: progRow.state
                                                        opacity: 0.75
                                                        font.pointSize: Math.max(8, root.font.pointSize - 2)
                                                        color: progRow.highlighted
                                                            ? root.palette.highlightedText
                                                            : root.palette.windowText
                                                    }
                                                }

                                                TerminalSplitButton {
                                                    Layout.alignment: Qt.AlignVCenter
                                                    visible: progRow.can_reattach
                                                    dropWidth: 24
                                                    primaryText: "Attach"
                                                    primaryToolTip: "Reattach in default terminal emulator"
                                                    dropToolTip: "Choose terminal emulator to reattach"
                                                    onTriggered: (binary) => backend.reattach_pty(progRow.launch_id, binary)
                                                }

                                                Button {
                                                    Layout.alignment: Qt.AlignVCenter
                                                    text: progRow.term_sent ? "Force Kill" : "Stop"
                                                    icon.name: "process-stop"
                                                    onClicked: backend.stop_program(progRow.launch_id)
                                                }
                                            }
                                        }
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        visible: runningAppsList.count === 0
                                        text: "No running applications."
                                        horizontalAlignment: Text.AlignHCenter
                                        opacity: 0.65
                                    }
                                }
                            }
                        }

                        // Settings popover button (gear icon + scale/microphone selectors)
                        ToolButton {
                            id: settingsBtn
                            activeFocusOnTab: true
                            icon.name: "settings-configure"
                            display: AbstractButton.IconOnly
                            Accessible.name: "Settings"
                            ToolTip.visible: hovered
                            ToolTip.delay: 400
                            ToolTip.text: "Settings (Ctrl+,) — Mic: " + backend.micSummary
                            onClicked: root.toggleSettingsPopup()
                            KeyNavigation.left: runningAppsBtn.visible && runningAppsBtn.enabled
                                ? runningAppsBtn
                                : (termSplit.enabled ? termSplit.dropButton : searchField)
                            KeyNavigation.tab: connectBtn.visible ? connectBtn : serverList
                            KeyNavigation.backtab: runningAppsBtn.visible && runningAppsBtn.enabled
                                ? runningAppsBtn
                                : (termSplit.enabled ? termSplit.dropButton : searchField)

                            Popup {
                                id: settingsPopup
                                // implicitWidth, not width: window-type popups resize to their
                                // implicit size when opened, which overrode an explicit width.
                                implicitWidth: Math.min(320, root.width - 32)
                                // Native popup window (xdg_popup on Wayland) so it can extend
                                // past the main window. Margins only matter if the platform
                                // falls back to an in-window Popup.Item.
                                popupType: Popup.Window
                                x: parent.width - width
                                y: parent.height + 4
                                margins: 16
                                padding: 12
                                modal: false
                                focus: true
                                closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent
                                onOpened: {
                                    Qt.callLater(function() {
                                        micCombo.forceActiveFocus()
                                    })
                                }
                                onClosed: settingsBtn.forceActiveFocus()

                                ColumnLayout {
                                    width: parent.width
                                    spacing: 8

                                    Label {
                                        text: "Render Scale Cap"
                                        font.bold: true
                                        Layout.fillWidth: true
                                    }

                                    WindowComboBox {
                                        id: globalScaleCombo
                                        Layout.fillWidth: true
                                        model: ["Uncapped", "2.0x", "1.5x", "1.0x"]
                                        currentIndex: backend.globalMaxScaleIndex
                                        onActivated: (idx) => backend.select_global_max_scale(idx)
                                    }

                                    ToolSeparator {
                                        Layout.fillWidth: true
                                        orientation: Qt.Horizontal
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        Label {
                                            text: "Host Microphone"
                                            font.bold: true
                                            Layout.fillWidth: true
                                        }
                                        ToolButton {
                                            id: refreshMicsBtn
                                            icon.name: "view-refresh"
                                            display: AbstractButton.IconOnly
                                            Accessible.name: "Refresh PipeWire sources"
                                            onClicked: backend.refresh_mics()

                                            // Dedicated tooltip: this button lives in settingsPopup's
                                            // window (see note on the shared ToolTip instance).
                                            ToolTip {
                                                popupType: Popup.Window
                                                visible: refreshMicsBtn.hovered
                                                delay: 400
                                                text: "Refresh PipeWire sources"
                                            }
                                        }
                                    }

                                    WindowComboBox {
                                        id: micCombo
                                        Layout.fillWidth: true
                                        model: root.micsModel
                                        textRole: "label"
                                        currentIndex: backend.selectedMicIndex
                                        onActivated: (idx) => backend.select_mic(idx)
                                    }
                                }
                            }
                        }
                    }

                    // Error banner (when applicable)
                    Frame {
                        Layout.fillWidth: true
                        visible: backend.hasError
                        padding: 8

                        RowLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                text: backend.errorText
                                color: "#e74c3c"
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }

                            Button {
                                text: "Retry"
                                visible: !backend.isConnected && !backend.isConnecting && backend.activeTarget.length > 0
                                onClicked: backend.connect_host(backend.activeTarget)
                            }
                        }
                    }

                    // Explicit command hint banner (>cmd / !cmd)
                    Frame {
                        Layout.fillWidth: true
                        visible: backend.isConnected && backend.isExplicitCommand
                        padding: 10

                        RowLayout {
                            anchors.fill: parent
                            spacing: 8

                            Label {
                                text: backend.explicitCommandText.length > 0
                                    ? "Press Enter to run remote command: " + backend.explicitCommandText
                                    : "Type a remote command after > or ! and press Enter"
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }

                            Button {
                                text: "Run"
                                enabled: backend.explicitCommandText.length > 0
                                onClicked: backend.launch_primary_action()
                            }
                        }
                    }

                    // Main Application Grid
                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        GridView {
                            id: appGrid
                            anchors.fill: parent
                            clip: true
                            visible: backend.isConnected && !backend.isExplicitCommand
                            model: root.appsModel
                            cellWidth: 112
                            cellHeight: 96
                            activeFocusOnTab: false
                            keyNavigationEnabled: true
                            highlightFollowsCurrentItem: true

                            readonly property int columns: Math.max(1, Math.floor(width / cellWidth))

                            onCountChanged: {
                                if (count > 0 && searchField.activeFocus) {
                                    currentIndex = 0
                                    positionViewAtBeginning()
                                }
                            }

                            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                            KeyNavigation.tab: termSplit.enabled
                                ? termSplit.primaryButton
                                : (runningAppsBtn.visible && runningAppsBtn.enabled ? runningAppsBtn : settingsBtn)
                            KeyNavigation.backtab: serverList

                            Keys.onUpPressed: (event) => {
                                if (currentIndex < columns) {
                                    root.focusSearch()
                                    event.accepted = true
                                } else {
                                    moveCurrentIndexUp()
                                    event.accepted = true
                                }
                            }
                            Keys.onDownPressed: (event) => {
                                if (currentIndex + columns < count) {
                                    moveCurrentIndexDown()
                                }
                                event.accepted = true
                            }
                            Keys.onLeftPressed: (event) => {
                                if (currentIndex % columns === 0) {
                                    root.focusServerList()
                                    event.accepted = true
                                } else {
                                    moveCurrentIndexLeft()
                                    event.accepted = true
                                }
                            }
                            Keys.onRightPressed: (event) => {
                                if (currentIndex + 1 < count) {
                                    moveCurrentIndexRight()
                                }
                                event.accepted = true
                            }
                            Keys.onReturnPressed: {
                                if (currentIndex >= 0) {
                                    backend.launch_app_at(currentIndex)
                                    root.focusSearch()
                                }
                            }
                            Keys.onEnterPressed: {
                                if (currentIndex >= 0) {
                                    backend.launch_app_at(currentIndex)
                                    root.focusSearch()
                                }
                            }
                            Keys.onEscapePressed: {
                                root.focusSearch()
                            }
                            Keys.onPressed: (event) => {
                                // Type-to-search / Backspace / Delete seamlessly returns to searchField
                                if (event.key === Qt.Key_Backspace) {
                                    var cur = backend.searchQuery
                                    if (cur.length > 0) {
                                        backend.set_search_query(cur.slice(0, -1))
                                    }
                                    root.focusSearch()
                                    event.accepted = true
                                } else if (event.key === Qt.Key_Delete) {
                                    backend.set_search_query("")
                                    root.focusSearch()
                                    event.accepted = true
                                } else if (event.text.length > 0
                                    && !(event.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier))
                                    && event.key !== Qt.Key_Tab
                                    && event.key !== Qt.Key_Backtab
                                    && event.key !== Qt.Key_Return
                                    && event.key !== Qt.Key_Enter
                                    && event.key !== Qt.Key_Escape) {
                                    backend.set_search_query(backend.searchQuery + event.text)
                                    root.focusSearch()
                                    event.accepted = true
                                }
                            }

                            delegate: ItemDelegate {
                                id: appTile
                                required property int index
                                required property string name
                                required property string generic_name
                                required property string icon_source

                                width: appGrid.cellWidth - 6
                                height: appGrid.cellHeight - 6
                                highlighted: (appGrid.activeFocus && appGrid.currentIndex === index)
                                    || (searchField.activeFocus && index === 0)
                                ToolTip.visible: hovered
                                ToolTip.delay: 500
                                ToolTip.text: generic_name.length > 0
                                    ? name + " — " + generic_name
                                    : name

                                onClicked: {
                                    appGrid.currentIndex = index
                                    backend.launch_app_at(index)
                                    searchField.forceActiveFocus()
                                }

                                contentItem: ColumnLayout {
                                    spacing: 4

                                    Item {
                                        Layout.alignment: Qt.AlignHCenter
                                        width: 48
                                        height: 48

                                        Image {
                                            anchors.fill: parent
                                            source: appTile.icon_source
                                            visible: appTile.icon_source.length > 0
                                            fillMode: Image.PreserveAspectFit
                                            smooth: true
                                            mipmap: true
                                            sourceSize.width: 48
                                            sourceSize.height: 48
                                        }

                                        Label {
                                            anchors.centerIn: parent
                                            visible: appTile.icon_source.length === 0
                                            text: appTile.name.length > 0 ? appTile.name.charAt(0).toUpperCase() : "?"
                                            font.pointSize: 18
                                            font.bold: true
                                            opacity: 0.5
                                        }
                                    }

                                    Label {
                                        Layout.fillWidth: true
                                        text: appTile.name
                                        horizontalAlignment: Text.AlignHCenter
                                        elide: Text.ElideRight
                                        maximumLineCount: 1
                                        color: appTile.highlighted
                                            ? root.palette.highlightedText
                                            : root.palette.windowText
                                    }
                                }
                            }
                        }

                        // Connecting / Disconnected / Empty state overlay
                        ColumnLayout {
                            anchors.centerIn: parent
                            spacing: 10
                            visible: !backend.isConnected || (backend.isConnected && !backend.isExplicitCommand && backend.appCount === 0)

                            BusyIndicator {
                                Layout.alignment: Qt.AlignHCenter
                                running: backend.isConnecting
                                visible: backend.isConnecting
                            }

                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                horizontalAlignment: Text.AlignHCenter
                                opacity: 0.75
                                text: {
                                    if (backend.isConnecting) {
                                        return "Connecting to " + backend.activeLabel + "…"
                                    }
                                    if (!backend.isConnected) {
                                        return backend.activeTarget.length > 0
                                            ? "Disconnected from " + backend.activeLabel
                                            : "Add a server using the ⚙ button in the sidebar to get started."
                                    }
                                    if (backend.searchQuery.trim().length > 0) {
                                        return "No matching apps. Press Enter to run \"" + backend.searchQuery.trim() + "\" directly."
                                    }
                                    return "Loading remote applications…"
                                }
                            }

                            Button {
                                Layout.alignment: Qt.AlignHCenter
                                text: "Connect"
                                visible: !backend.isConnected && !backend.isConnecting && backend.activeTarget.length > 0
                                onClicked: backend.connect_host(backend.activeTarget)
                            }
                        }
                    }

                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Manage Connections Dialog (master-detail layout)
    // -----------------------------------------------------------------------
    Dialog {
        id: manageDialog
        title: "Manage Connections"
        // Real top-level dialog window (Qt::Dialog, transient for the main window; an
        // xdg_toplevel dialog on Wayland) rather than an overlay inside the main window. The
        // title goes to the window decoration; the KDE style hides its in-window header.
        popupType: Popup.Window
        modal: true
        // Positions the window over the main window where the platform allows it (X11);
        // on Wayland the compositor places transient dialogs itself.
        anchors.centerIn: parent
        // Its own window, so not tied to the main window's size. Set as implicit size: a
        // window-type popup re-sizes itself to its implicit size when its content is reparented
        // into the window on open(), which overrode an explicit `height`. If the platform falls
        // back to an in-window popup, the style's margins shrink it to fit.
        implicitWidth: 680
        implicitHeight: 440
        standardButtons: Dialog.Close

        property string selectedTarget: ""
        property bool isNewHost: true

        function startNewHost() {
            selectedTarget = ""
            isNewHost = true
            labelField.text = ""
            targetField.text = ""
            portSpin.value = 7443
            serverBinField.text = ""
            audioCheck.checked = true
            hostScaleCombo.currentIndex = 0
            labelField.forceActiveFocus()
        }

        function loadHost(label, sshTarget, port, serverBin, audioEnabled, maxScaleIndex) {
            selectedTarget = sshTarget
            isNewHost = false
            labelField.text = label
            targetField.text = sshTarget
            portSpin.value = port > 0 ? port : 7443
            serverBinField.text = serverBin
            audioCheck.checked = audioEnabled
            hostScaleCombo.currentIndex = maxScaleIndex >= 0 ? maxScaleIndex : 0
        }

        onOpened: startNewHost()

        RowLayout {
            anchors.fill: parent
            spacing: 12

            // Left pane: Saved servers list + New Server button
            ColumnLayout {
                Layout.preferredWidth: 215
                Layout.fillHeight: true
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Saved Servers"
                        font.bold: true
                        Layout.fillWidth: true
                    }
                    Button {
                        text: "New"
                        icon.name: "list-add"
                        onClicked: manageDialog.startNewHost()
                    }
                }

                Frame {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    padding: 2

                    ListView {
                        id: manageHostList
                        anchors.fill: parent
                        clip: true
                        model: root.hostsModel
                        spacing: 2

                        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                        delegate: ItemDelegate {
                            id: mgmtRow
                            required property string label
                            required property string ssh_target
                            required property int port
                            required property string server_bin
                            required property bool audio_enabled
                            required property int max_scale_index

                            width: manageHostList.width
                            highlighted: !manageDialog.isNewHost && manageDialog.selectedTarget === ssh_target
                            onClicked: manageDialog.loadHost(label, ssh_target, port, server_bin, audio_enabled, max_scale_index)

                            contentItem: ColumnLayout {
                                spacing: 1
                                Label {
                                    text: mgmtRow.label
                                    font.bold: mgmtRow.highlighted
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                    color: mgmtRow.highlighted
                                        ? root.palette.highlightedText
                                        : root.palette.windowText
                                }
                                Label {
                                    text: mgmtRow.ssh_target
                                    opacity: 0.7
                                    font.pointSize: Math.max(8, root.font.pointSize - 2)
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                    color: mgmtRow.highlighted
                                        ? root.palette.highlightedText
                                        : root.palette.windowText
                                }
                            }
                        }

                        Label {
                            anchors.centerIn: parent
                            visible: manageHostList.count === 0
                            text: "No saved servers"
                            opacity: 0.6
                        }
                    }
                }
            }

            ToolSeparator {
                Layout.fillHeight: true
            }

            // Right pane: Server editor form
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 10

                Label {
                    text: manageDialog.isNewHost ? "Add New Server" : "Edit Server"
                    font.bold: true
                    font.pointSize: root.font.pointSize + 1
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 10
                    rowSpacing: 8

                    Label {
                        text: "Display Name"
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                    }
                    TextField {
                        id: labelField
                        Layout.fillWidth: true
                        placeholderText: "e.g. Workstation"
                    }

                    Label {
                        text: "SSH Target"
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                    }
                    TextField {
                        id: targetField
                        Layout.fillWidth: true
                        placeholderText: "user@hostname"
                    }

                    Label {
                        text: "QUIC Port"
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                    }
                    SpinBox {
                        id: portSpin
                        from: 1
                        to: 65535
                        value: 7443
                        editable: true
                    }

                    Label {
                        text: "Server Binary"
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                    }
                    TextField {
                        id: serverBinField
                        Layout.fillWidth: true
                        placeholderText: "remotewayland-server (optional)"
                    }

                    Label {
                        text: "Render Scale Cap"
                        Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                    }
                    WindowComboBox {
                        id: hostScaleCombo
                        Layout.fillWidth: true
                        model: ["Use Global Default", "Uncapped", "2.0x", "1.5x", "1.0x"]
                        currentIndex: 0
                        onActivated: (idx) => currentIndex = idx
                    }

                    Item { implicitWidth: 1; implicitHeight: 1 }
                    CheckBox {
                        id: audioCheck
                        text: "Enable PipeWire audio bridge"
                        checked: true
                    }
                }

                Item { Layout.fillHeight: true }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Button {
                        text: "Delete"
                        icon.name: "edit-delete"
                        enabled: !manageDialog.isNewHost && manageDialog.selectedTarget.length > 0
                        onClicked: {
                            backend.delete_host(manageDialog.selectedTarget)
                            manageDialog.startNewHost()
                        }
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: manageDialog.isNewHost ? "Add Server" : "Save Changes"
                        icon.name: manageDialog.isNewHost ? "list-add" : "document-save"
                        enabled: targetField.text.trim().length > 0
                        highlighted: true
                        onClicked: {
                            var savedTarget = targetField.text.trim()
                            var savedLabel = labelField.text.trim().length > 0
                                ? labelField.text.trim()
                                : savedTarget
                            backend.add_or_update_host(
                                manageDialog.isNewHost ? "" : manageDialog.selectedTarget,
                                savedLabel,
                                savedTarget,
                                serverBinField.text,
                                portSpin.value,
                                audioCheck.checked,
                                hostScaleCombo.currentIndex
                            )
                            manageDialog.selectedTarget = savedTarget
                            manageDialog.isNewHost = false
                        }
                    }
                }
            }
        }
    }
}
