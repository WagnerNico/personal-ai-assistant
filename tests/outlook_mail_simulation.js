// Execute the production reader against getter-only Outlook objects.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

function mailbox({running = true, denied = false, changed = false, accounts = 1} = {}) {
    const rows = [
        {id: 10, read: false, date: '2026-10-09T12:00:00Z', body: 'Older message'},
        {id: 11, read: true, date: '2026-10-11T13:00:00Z', body: 'Already read'},
        {id: 12, read: false, date: '2026-10-11T12:00:00Z', body: 'x'.repeat(2100)},
        {id: 13, read: false, date: '2026-10-10T12:00:00Z', body: 'Middle message'},
    ];
    const before = rows.map(row => row.read);
    const bodyReads = [];
    let idCalls = 0;
    const inbox = {unreadCount: () => rows.filter(row => !row.read).length,
        incomingMessages: {
            whose: filter => {
                assert.deepEqual(JSON.parse(JSON.stringify(filter)), {isRead: false});
                const unread = () => rows.filter(row => !row.read);
                return {id: () => (++idCalls === 2 && changed ? [] : unread().map(row => row.id)),
                        timeReceived: () => unread().map(row => new Date(row.date))};
            },
            byId: id => {
                const row = rows.find(item => item.id === id);
                assert.equal(row.read, false);
                return Object.freeze({isRead: () => row.read,
                    sender: () => ({name: 'Test sender', address: 'sender@example.test'}),
                    subject: () => 'Subject ' + id,
                    plainTextContent: () => {bodyReads.push(id); return row.body;}});
            }
        }};
    const context = {Application: id => {
        assert.equal(id, 'com.microsoft.Outlook');
        return Object.freeze({running: () => running, exchangeAccounts: () => Array(accounts).fill({}),
            imapAccounts: () => [], popAccounts: () => [], get inbox() {
            if (denied) throw {number: -1743};
            return inbox;
        }});
    }};
    vm.createContext(context);
    vm.runInContext(source, context);
    const result = args => JSON.parse(context.run(args));
    return {result, bodyReads, unchanged: () => assert.deepEqual(rows.map(row => row.read), before)};
}

const reader = mailbox();
const data = reader.result(['unread', '2', 'body']);
assert.equal(data.unread_count, 3);
assert.deepEqual(data.messages.map(row => row.id), [12, 13]);
assert.equal(data.messages[0].body.length, 2000);
assert.equal(data.messages[0].body_truncated, true);
assert.ok(data.messages.every(row => row.unread === true));
assert.deepEqual(reader.bodyReads, [12, 13]);
reader.unchanged();
const headers = mailbox();
assert.equal(headers.result(['unread', '10', 'headers']).messages.length, 3);
assert.deepEqual(headers.bodyReads, []);
headers.unchanged();
assert.equal(mailbox({running: false}).result(['unread']).error, 'not_running');
assert.equal(mailbox({denied: true}).result(['unread']).error, 'automation');
assert.equal(mailbox({changed: true}).result(['unread']).error, 'mailbox_changed');
assert.equal(mailbox({accounts: 0}).result(['unread']).error, 'mailbox_unavailable');
assert.throws(() => reader.result(['mark_read']));
assert.throws(() => reader.result(['unread', '99', 'body']));
