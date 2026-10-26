// Run the real helper's state transitions with fake OS state and no media events.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8');
const item = {bundle: 'test.player', playing: true, title: 'Track', item: '1'};

function simulate(action, before, token, options = {}) {
    let current = before ? {...before} : null;
    const commands = [];
    const context = vm.createContext({
        ObjC: {import() {}, bindFunction() {}},
        $: {
            NSBundle: {bundleWithPath() { return {load: true}; }},
            NSRunLoop: {currentRunLoop: {runUntilDate() {}}},
            NSDate: {dateWithTimeIntervalSinceNow() {}},
            NSDictionary: {dictionary: {}},
            MRMediaRemoteSendCommand(command, optionsDictionary) {
                assert.equal(typeof optionsDictionary, 'object');
                assert.notEqual(optionsDictionary, null);
                commands.push(command);
                if (options.respond !== false) current.playing = command === 0;
                if (options.changeItem) current.title = 'Another track';
            },
        },
        readState: () => current ? {...current} : null,
    });
    vm.runInContext(source, context);
    vm.runInContext('state = readState;', context);
    const result = JSON.parse(context.run([action, JSON.stringify(token)]));
    return {result, commands};
}

assert.deepEqual(simulate('pause', null), {result: {paused: false}, commands: []});
assert.deepEqual(simulate('pause', {...item, playing: false}), {result: {paused: false}, commands: []});
const paused = simulate('pause', item);
assert.equal(paused.result.paused, true);
assert.deepEqual(paused.commands, [1]);
assert.deepEqual(paused.result.token, item);
const resumed = simulate('resume', {...item, playing: false}, item);
assert.equal(resumed.result.resumed, true);
assert.deepEqual(resumed.commands, [0]);
for (const before of [null, item, {...item, bundle: 'other.player', playing: false},
                      {...item, title: 'Other track', playing: false},
                      {...item, item: '2', playing: false}]) {
    const skipped = simulate('resume', before, item);
    assert.equal(skipped.result.skipped, true);
    assert.deepEqual(skipped.commands, []);
}
const failure = simulate('pause', item, null, {respond: false});
assert.equal(failure.result.paused, false);
assert.equal(failure.result.token, null);
assert.equal(failure.result.needsAccessibility, true);
assert.deepEqual(failure.commands, [1]);
const changed = simulate('pause', item, null, {changeItem: true});
assert.equal(changed.result.paused, false);
assert.equal(changed.result.token, null);
assert.equal(simulate('wait-paused', {...item, playing: false}, item).result.changed, true);
assert.equal(simulate('wait-resumed', item, item).result.changed, true);
assert.equal(simulate('wait-paused', {...item, title: 'Another track', playing: false}, item).result.changed, false);
console.log('Media helper ownership and state transitions passed.');
