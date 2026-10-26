"""Recognize explicit music requests while leaving discussion and quotes in chat."""

import re

from actions.spotify import SpotifyRequest, SpotifyResult, track_uri


def spotify_request(text, *, prefix):
    command = text.strip().rstrip('.!?')
    command = re.sub(r'(?:,?\s+(?:please|for me|bitte))+$', '', command, flags=re.I)
    match = re.fullmatch(prefix + r'(?P<verb>play|resume|continue|pause|stop|spiele|spiel|pausiere|pausier|stoppe|starte)\s+(?P<item>.+)',
                         command, re.I | re.S)
    if not match:
        match = re.fullmatch(r'^(?:airis,?\s+)?(?:kannst du|könntest du|würdest du|bitte)\s+'
                             r'(?:bitte\s+)?(?P<item>.+?)\s+(?P<verb>abspielen|spielen|pausieren|anhalten|fortsetzen)',
                             command, re.I | re.S)
    if not match:
        return None
    explicit_spotify = bool(re.search(r'\s+(?:on|in|auf|bei)\s+spotify$', match['item'], re.I))
    item = re.sub(r'\s+(?:on|in|auf|bei)\s+spotify$', '', match['item'], flags=re.I).strip()
    if match['verb'].casefold() in ('spiele', 'spiel', 'pausiere', 'pausier', 'stoppe', 'starte'):
        item = re.sub(r'^mal\s+', '', item, flags=re.I)
    if match['verb'].casefold() in ('spiele', 'spiel'):
        item = re.sub(r'\s+ab$', '', item, flags=re.I)
    pause = match['verb'].casefold() in ('pause', 'stop', 'pausiere', 'pausier', 'stoppe', 'pausieren', 'anhalten')
    generic = re.fullmatch(r'(?:spotify|(?:(?:the|my|some|die|meine|etwas)\s+)?(?:music|musik))', item, re.I)
    if match['verb'].casefold() == 'starte' and item.casefold() == 'spotify':
        return None  # Preserve the existing app-opening command.
    if pause and not generic:
        return None
    if generic:
        return SpotifyRequest('pause' if pause else 'play')
    uri = track_uri(item)
    if (not explicit_spotify and not uri and
            not re.search(r'\b(?:by|von|song|track)\b', item, re.I)):
        return None
    if match['verb'].casefold() in ('resume', 'continue', 'starte', 'fortsetzen'):
        return None
    if re.search(r'[;\n&]|\b(?:and|then|und|danach)\s+(?:open|play|pause|delete|send|close|skip|öffne|spiele|pausiere|lösche|sende|schließe)\b', item, re.I):
        return SpotifyResult("Please request one Spotify playback action at a time.")
    if uri:
        return SpotifyRequest('play', uri=uri)
    if re.search(r'://|spotify:|\b(?:playlist|album)\b', item, re.I):
        return SpotifyResult("Please name a song or artist, or provide a Spotify track link.")
    try:
        return SpotifyRequest('play', query=item.strip('"\''))
    except ValueError:
        return SpotifyResult("Please use a shorter song or artist name.")
