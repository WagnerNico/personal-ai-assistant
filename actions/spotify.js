// Fixed Spotify scripting commands. Track URIs are data, never executable code.
ObjC.import('AppKit');

function snapshot(spotify) {
    const state = spotify.playerState();
    let title = '', artist = '', uri = '';
    try {
        const track = spotify.currentTrack;
        title = track.name();
        artist = track.artist();
        uri = track.id();
    } catch (_) {}
    return {playing: state === 'playing', title: title, artist: artist, uri: uri};
}

function wait() {
    $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(0.1));
}

function run(argv) {
    const action = argv[0], uri = argv[1] || '';
    if (!['play', 'pause', 'status'].includes(action)) throw new Error('Unsupported Spotify action.');
    if (uri && (action !== 'play' || !/^spotify:track:[A-Za-z0-9]{22}$/.test(uri))) {
        throw new Error('Only a Spotify track URI can be played.');
    }
    const spotify = Application('com.spotify.client');
    if (action === 'play') {
        for (let i = 0; i < 50 && !spotify.running(); i++) wait();
    }
    if (!spotify.running()) {
        if (action === 'pause' || action === 'status') return JSON.stringify({ok: true, running: false, playing: false});
        return JSON.stringify({ok: false, error: 'not_running'});
    }
    try {
        if (action === 'status') return JSON.stringify(Object.assign({ok: true, running: true}, snapshot(spotify)));
        if (action === 'pause') spotify.pause();
        else if (uri) spotify.playTrack(uri);
        else spotify.play();
        let after, stable = 0;
        for (let i = 0; i < 80; i++) {
            after = snapshot(spotify);
            if (after.playing === (action === 'play') && (!uri || after.uri === uri)) {
                stable++;
                if (stable >= 3) return JSON.stringify(Object.assign({ok: true, running: true}, after));
            } else {
                stable = 0;
                // A track-loading operation can overtake an earlier pause.
                // Reapply pause only; repeating playTrack would restart music.
                if (action === 'pause' && i % 6 === 5) spotify.pause();
            }
            wait();
        }
        return JSON.stringify({ok: false, error: 'not_confirmed'});
    } catch (error) {
        return JSON.stringify({ok: false, error: Number(error.number) === -1743 ? 'automation' : 'spotify_error'});
    }
}
