// Exercise the real Spotify helper using an isolated application simulation.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8');
const uri = 'spotify:track:4iV5W9uYEdYUVa79Axb7Rh';

function simulate(action, requested = '', options = {}) {
    let playing = Boolean(options.playing);
    const calls = [];
    const app = {
        running: () => options.running !== false,
        playerState: () => playing ? 'playing' : 'paused',
        currentTrack: {name: () => 'Hello', artist: () => 'Adele', id: () => uri},
        play: () => { calls.push('play'); if (options.respond !== false) playing = true; },
        pause: () => {
            calls.push('pause');
            if (options.respond !== false && (!options.latePause || calls.length > 1)) playing = false;
        },
        playTrack: value => { assert.equal(value, uri); calls.push('track'); playing = true; },
    };
    if (options.denied) app.play = () => { const error = new Error('Denied'); error.number = -1743; throw error; };
    const context = vm.createContext({
        ObjC: {import() {}},
        $: {NSRunLoop: {currentRunLoop: {runUntilDate() {}}}, NSDate: {dateWithTimeIntervalSinceNow() {}}},
        Application: bundle => { assert.equal(bundle, 'com.spotify.client'); return app; },
    });
    vm.runInContext(source, context);
    return {result: JSON.parse(context.run([action, requested])), calls};
}
assert.equal(simulate('play').result.playing, true);
assert.equal(simulate('pause', '', {playing: true}).result.playing, false);
assert.deepEqual(simulate('pause', '', {running: false}).calls, []);
assert.equal(simulate('play', uri).result.uri, uri);
assert.equal(simulate('play', '', {denied: true}).result.error, 'automation');
assert.equal(simulate('play', '', {respond: false}).result.ok, false);
const latePause = simulate('pause', '', {playing: true, latePause: true});
assert.equal(latePause.result.ok, true);
assert.deepEqual(latePause.calls, ['pause', 'pause']);
assert.throws(() => simulate('play', 'file:///private/data'));
assert.throws(() => simulate('quit'));
console.log('Spotify playback, pause, verification and scope passed.');
