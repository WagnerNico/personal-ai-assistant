// Fixed macOS helper: observe Now Playing and pause/resume that same item only.
ObjC.import('Foundation');
ObjC.import('AppKit');
$.NSBundle.bundleWithPath('/System/Library/PrivateFrameworks/MediaRemote.framework').load;
ObjC.bindFunction('MRMediaRemoteSendCommand', ['bool', ['int', 'id']]);

function stringValue(value) {
    return !value || value.isNil() ? '' : ObjC.unwrap(value.description);
}

function state() {
    const request = $.NSClassFromString('MRNowPlayingRequest');
    const item = request.localNowPlayingItem;
    if (!item || item.isNil()) return null;
    const client = request.localNowPlayingPlayerPath.client;
    const bundle = stringValue(client.bundleIdentifier);
    if (!bundle || $.NSRunningApplication.runningApplicationsWithBundleIdentifier(bundle).count === 0) return null;
    const info = item.nowPlayingInfo;
    return {bundle: bundle, playing: Boolean(request.localIsPlaying),
            title: stringValue(info.objectForKey('kMRMediaRemoteNowPlayingInfoTitle')),
            item: stringValue(info.objectForKey('kMRMediaRemoteNowPlayingInfoUniqueIdentifier'))};
}

function sameItem(a, b) {
    return a && b && a.bundle === b.bundle && a.title === b.title && a.item === b.item;
}

function waitForChange(before, playing) {
    for (let i = 0; i < 25; i++) {
        const now = state();
        if (!sameItem(before, now) || now.playing === playing) return now;
        $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(0.04));
    }
    return state();
}

function run(argv) {
    const action = argv[0];
    const before = state();
    if (action === 'status') return JSON.stringify(before);
    // Python posts media keys through native AppKit; JXA's opaque CGEvent bridge
    // does not preserve the event pointer. These actions only verify the result.
    if (action === 'wait-paused' || action === 'wait-resumed') {
        const token = JSON.parse(argv[1]);
        const playing = action === 'wait-resumed';
        const after = waitForChange(token, playing);
        return JSON.stringify({changed: sameItem(token, after) && after.playing === playing});
    }
    if (!['pause', 'resume'].includes(action)) throw new Error('Unsupported media action.');
    const resume = action === 'resume';
    if (!before) return JSON.stringify(resume ? {resumed: false, skipped: true} : {paused: false});
    if (resume && !sameItem(before, JSON.parse(argv[1]))) return JSON.stringify({resumed: false, skipped: true});
    if (before.playing === resume) return JSON.stringify(resume ? {resumed: false, skipped: true} : {paused: false});
    // An explicit command is the fallback without Accessibility. Never toggle
    // here: the Python caller owns the one native-key attempt for this turn.
    // JXA maps JavaScript null to NSNull for an id argument; the native API
    // expects a dictionary and copies it. Empty options avoid that conversion.
    $.MRMediaRemoteSendCommand(resume ? 0 : 1, $.NSDictionary.dictionary);
    const after = waitForChange(before, resume);
    const changed = sameItem(before, after) && after.playing === resume;
    const needsAccessibility = !changed;
    return JSON.stringify(resume ? {resumed: changed, needsAccessibility: needsAccessibility} :
        {paused: changed, token: changed ? before : null, needsAccessibility: needsAccessibility});
}
