// Read-only Outlook access. Never open messages, select them, or set properties.

function run(argv) {
    const action = argv[0];
    if (!['status', 'unread'].includes(action)) throw new Error('Unsupported read-only Outlook action.');
    const limit = Number(argv[1] || 5);
    const includeBody = argv[2] === 'body';
    if (!Number.isInteger(limit) || limit < 1 || limit > 10 ||
        (argv[2] && !['body', 'headers'].includes(argv[2])))
        throw new Error('Invalid read-only Outlook request.');
    const outlook = Application('com.microsoft.Outlook');
    if (!outlook.running()) return JSON.stringify({ok: false, error: 'not_running'});
    try {
        const inbox = outlook.inbox;
        const count = inbox.unreadCount();
        if (!Number.isInteger(count) || count < 0) return JSON.stringify({ok: false, error: 'unsupported'});
        const accounts = outlook.exchangeAccounts().length + outlook.imapAccounts().length + outlook.popAccounts().length;
        if (accounts === 0)
            return JSON.stringify({ok: false, error: 'mailbox_unavailable'});
        if (action === 'status')
            return JSON.stringify({ok: true, unread_count: count, scope: 'default_inbox',
                                   inbox_message_count: inbox.incomingMessages.id().length,
                                   account_count: accounts});
        if (count === 0)
            return JSON.stringify({ok: true, unread_count: 0, scope: 'default_inbox', messages: []});
        if (count > 10000) return JSON.stringify({ok: false, error: 'too_many'});
        const unread = inbox.incomingMessages.whose({isRead: false});
        // Bulk metadata avoids fetching every email body just to find the newest.
        const ids = unread.id();
        const dates = unread.timeReceived();
        const checkedIds = unread.id();
        if (JSON.stringify(ids) !== JSON.stringify(checkedIds) || ids.length !== dates.length)
            return JSON.stringify({ok: false, error: 'mailbox_changed'});
        if (ids.length > 10000 || ids.length !== count)
            return JSON.stringify({ok: false, error: 'mailbox_changed'});
        const newest = ids.map((id, index) => ({id, date: new Date(dates[index])}));
        if (newest.some(item => !Number.isInteger(item.id) || !Number.isFinite(item.date.getTime())))
            return JSON.stringify({ok: false, error: 'unsupported'});
        newest.sort((a, b) => b.date - a.date || b.id - a.id);
        const messages = [];
        for (const item of newest.slice(0, limit)) {
            const message = inbox.incomingMessages.byId(item.id);
            if (message.isRead() !== false)
                return JSON.stringify({ok: false, error: 'mailbox_changed'});
            const sender = message.sender();
            const subject = message.subject();
            const content = includeBody ? message.plainTextContent() : '';
            if (typeof subject !== 'string' || typeof content !== 'string' || !sender)
                return JSON.stringify({ok: false, error: 'unsupported'});
            messages.push({id: item.id, subject: subject.slice(0, 300),
                           sender: String(sender.name || sender.address || 'Unknown sender').slice(0, 300),
                           received_at: item.date.toISOString(), unread: true,
                           body: content.slice(0, 2000), body_truncated: content.length > 2000});
            // Getters leave flags alone. Never try to restore a concurrently changed flag.
            if (message.isRead() !== false)
                return JSON.stringify({ok: false, error: 'mailbox_changed'});
        }
        return JSON.stringify({ok: true, unread_count: count, scope: 'default_inbox', messages});
    } catch (error) {
        return JSON.stringify({ok: false, error: Number(error.number) === -1743 ? 'automation' : 'unsupported'});
    }
}
