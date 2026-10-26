# Airis

Offline synthetic wake-word corpus generation, custom detector training, streaming
evaluation, and current limitations are documented in
[training/wakeword/README.md](training/wakeword/README.md). Trained prototypes are
stored locally. The desktop UI offers optional experimental wake-word listening;
it is disabled by default and has not passed the recognition accuracy targets.

## Desktop conversation and push-to-talk

Airis uses one shared virtual environment, `.venv`, for the desktop interface,
Qwen processing, Whisper, and MLX speech generation. To set up or update it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -u -m ui.desktop
```

For daily use without a terminal, double-click **Airis.app** in this project
folder. You can keep it in the Dock. The launcher uses `.venv`
and records startup errors in `logs/desktop.log`. Keep the project and environment
in their current location; this launcher is not a standalone packaged product.
Rebuild it after moving the project with `python -m ui.build_launcher`.
On Apple silicon, the launcher requests native ARM execution; Rosetta is not
required when using the ARM-native project environment.

The window opens immediately and prepares Qwen in the background before accepting
messages. Startup loads the cached model into memory and runs a short warm-up to
compile the first generation path. The progress indicator and status show when
the session is ready. **Prepare / Retry** retries preparation after an error.
The model stays loaded until you quit Airis or choose another model.

Closing the main window keeps Airis running in the background. Use the Airis
menu-bar icon to **Open Airis**, toggle **Enable wake word**, or **Quit Airis**.
The standard **Quit Airis** menu action and Command + Q also stop the app.
Explicit quit waits for active playback or inference to finish, then releases
the microphone, models, and menu-bar icon. Relaunching Airis opens the existing
window rather than loading a second copy of the models. Quit an older running
version before launching this version for the first time.

The sidebar has three areas:

- **Settings**: select the response model, voice, language, and maximum reply tokens.
  Enable optional wake-word listening and show its test controls.
  **Prepare / Retry** prepares the runtime or retries after an error.
- **Text chat**: write and read messages with silent text replies. Press Enter to
  send, or Shift + Enter to add a new line.
- **Voice chat**: hold the large talk button while speaking, then release it.
  Airis transcribes your recording and streams its reply aloud. **Show transcript**
  reveals the conversation text when needed.

The chat areas share one session and conversation history. Switching areas does
not clear the conversation. Settings and the selected area persist between launches.

### Opening, switching and full screen

Type or say **Open Outlook**, **Open Spotify**, or **Open Outlook and make it
full screen**. The same commands work in Text chat, Voice chat, and background
wake-word conversations. Airis brings the requested installed app forward and
returns a short confirmation in the selected reply mode. Outlook resolves to
**Microsoft Outlook**. Apps are discovered in `/Applications`, `~/Applications`,
and the standard macOS system app folders, including Utilities.

Type or say **Switch to Spotify**, **Go to Outlook**, or **Please switch back to
WhatsApp** to bring that app forward. If it is closed, Airis opens it first.
These commands work in all three conversation modes. Switching reuses the
existing app instance and confirms that the requested app has become active,
waiting up to 20 seconds for initial launch if necessary. You can also say
**Switch to Spotify in full screen** to request both actions.

Full screen also works as a standalone command: **Make Spotify full screen**,
**Put Outlook in full screen**, or **Set WhatsApp to full screen**. Airis targets
the named running app without launching or switching apps first. If the app is
closed, it asks you to open it first. Standalone commands use the same window
readiness checks and full-screen confirmation as combined commands, and work
in Text chat, Voice chat and background wake conversations.

Opening and switching apps need no Accessibility permission. Full screen uses the macOS
Accessibility API; the first explicit full-screen request shows the system's
permission prompt if needed. Allow the launching **Airis** or **Python** process
in **System Settings > Privacy & Security > Accessibility**, then repeat the
command. macOS may require quitting and reopening Airis after granting access.
The local launcher currently runs Python rather than a standalone signed app,
so the permission entry can be named Python or your terminal. Apps with windows
that do not expose full-screen control remain open and Airis reports the limitation.
An already full-screen window stays full screen.
Full-screen requests have two separate stages after launching the app. Airis
first waits up to 20 seconds for the app to finish launching and expose the same
usable window on two consecutive checks. Only then does it request full screen,
with a separate 12-second confirmation budget. The UI displays **Waiting for
the app to finish opening** followed by **Making the app full screen**.
Confirmation reads the app's current windows, including replacements created
during the transition. If an acknowledged full-screen setting has no effect,
Airis uses the requested app's explicit **Enter Full Screen** menu action or its
dedicated full-screen button, then verifies the result. It never presses an Exit
command, maximizes via the zoom button, or sends shortcuts to another app.
Persistent failures include their stage and reason in `logs/desktop.log` when
using the local launcher.

The tool interface supports opening or switching to one app with optional full
screen, and setting a running app to full screen as a separate action.
This app-opening tool does not send messages, play music, close apps, open documents or websites,
or run generated shell commands. Additional actions in the same request are
rejected. App names must match installed bundles; ambiguous matches require a
full name. Ordinary discussion, quoted examples and negated commands do not
execute tools.

`processing/agent.py` handles direct commands without model generation for a
quick response. Varied command wording can use the response model to select
a structured `open_app`, `switch_app` or `full_screen_app` request, restricted
to the requested action and apps named in the current message.
Plans never enter speech output or history; only the actual tool result does.
The reusable interface in `actions/app_control.py` is
`AppControl.execute(AppRequest("Outlook", full_screen=True))` or
`AppControl.execute(AppRequest("Spotify", action="switch_app"))` for activation.
For full screen without launching, use
`AppControl.execute(AppRequest("Spotify", full_screen=True, action="full_screen_app"))`.
Inject `app_control=` into `Assistant` to replace the
tool service, or set `app_control=False` to disable local actions for an embedded
client. `processing.process_text` remains a text-only model API.

### Spotify playback

Airis controls the installed macOS Spotify app through its scripting interface.
No Spotify developer app, OAuth token or Web API credentials are needed. Sign
in to Spotify first. These commands work in Text chat, Voice chat, and background
voice sessions:

- **Play Spotify**, **Resume the music**, or **Play music on Spotify** resumes
  Spotify's current song or queue, opening the app in the background if needed.
- **Pause Spotify** or **Pause the music**
  pauses Spotify without launching it if it is closed.
- **Play Hello by Adele on Spotify** finds an indexed Spotify track and plays it.
  Song-name lookup requires **Enable web search and page reading**. Airis accepts
  only Spotify track links matching the requested song/artist from the results;
  it does not load Spotify pages or ask the LLM to invent a track identifier.
- **Play https://open.spotify.com/track/TRACK_ID** or **Play spotify:track:TRACK_ID**
  plays a direct track link or URI without web search. Replace `TRACK_ID` with
  the 22-character ID from the track's real Spotify link.

macOS may ask permission for Airis or its launcher to control Spotify. Allow it
in **System Settings > Privacy & Security > Automation**. This is separate from
Accessibility access. Airis confirms the actual Spotify playback state before
reporting success; a missing app, blocked automation or unavailable playback is
reported in the conversation. It does not send global media-key toggles for these
commands. Spoken confirmations are brief; Spotify can start during the confirmation.
Playback must match on three consecutive checks. If track loading overtakes a
Pause command, Airis reapplies Pause within a bounded confirmation window.

An explicit Pause stays paused when a speech turn or background voice session
ends. An explicit Play replaces automatic resume ownership for the previous
media item, so closing the session cannot restart an old song or browser video.
The next held voice recording can still temporarily pause the newly playing music.

Only playback/resume, pause and individual track selection are exposed. Playlist
editing, downloads, volume changes and account changes are outside this tool.
The reusable interface is `SpotifyControl.execute(SpotifyRequest("play"))` or
`SpotifyControl.execute(SpotifyRequest("pause"))` in `actions/spotify.py`.
Inject `spotify_control=` into `Assistant` to replace it, or set it to `False`
to disable it. `app_control=False` also disables Spotify by default for embedded
clients. UI integrations use `on_music_result` to respect explicit playback intent
when managing temporary speech interruptions.

### Read-only email access (Gmail, including Gmail in Outlook)

Your Outlook app can display Gmail emails without exposing them to macOS scripts.
Airis therefore connects directly to **Gmail**, using only the OAuth permission
`https://www.googleapis.com/auth/gmail.readonly`. This permission cannot send,
modify, delete, or mark messages read. The Gmail account should be the same one
you use in Outlook. Outlook does not need to be open.

Restart Airis after updating. **Settings > Email (read only)** has an access
switch and **Connect Gmail** / **Disconnect Gmail** buttons. This applies to
Text chat, Voice chat, and background voice sessions. Try:

- **Read my latest unread emails.** Summarizes the five newest unread messages.
- **Summarize my latest 3 unread emails.** Requests up to ten messages per turn.
- **List my unread emails.** Lists senders, subjects and received timestamps
  without fetching message bodies.
- **Do I have any new emails?** Checks unread Inbox messages on demand.

#### One-time Google connection for development

Google requires an OAuth client for the app and your explicit sign-in. Follow
[Google’s Gmail desktop setup](https://developers.google.com/workspace/gmail/api/quickstart/python):

1. Create or select a project in Google Cloud and **enable the Gmail API**.
2. Configure **Google Auth platform > Branding** for Airis. For a personal Gmail
   account, select **External** under Audience, keep the app in testing, and add
   your Gmail address as a test user. A Workspace account may use Internal if
   your organization permits it.
3. Under **Data Access**, add only the Gmail read-only scope above. Under
   **Clients**, create an OAuth client with application type **Desktop app**
   and download its JSON file. Keep this file outside the repository.
4. In Airis, click **Connect Gmail**, choose that downloaded JSON, and complete
   Google sign-in in your browser. Choose the same account as in Outlook and
   approve read-only Gmail access. Airis waits up to three minutes for sign-in.
5. Ask **Read my latest unread emails**. You can then check in Outlook that
   those messages remain unread.

OAuth credentials and refresh tokens are stored in **macOS Keychain**, never in
`token.json`, QSettings, the repository, or model prompts. The reader refreshes
access silently when possible; Google testing-mode grants may expire and require
reconnection. Disconnect removes the local Keychain entry; to revoke the grant
at Google too, remove Airis from your Google account’s connected apps.
For public distribution, provide Airis’s own OAuth client and complete Google’s
requirements for the [restricted Gmail read-only scope](https://developers.google.com/workspace/gmail/api/auth/scopes).
Users of the eventual product should not need to create their own OAuth client.

#### Read-only boundaries

The reader covers the connected account’s **Gmail Inbox**, including every Inbox
category, with the **UNREAD** label. Already-read mail, other folders, attachments,
and unsolicited polling are outside this initial feature. Requests use fixed
Gmail GET endpoints only. No message selection, read-flag writes, sending, replies,
or modifications are implemented. Concurrent changes by another mail client can
still change a message’s status independently of Airis.

Email contents are untrusted data and cannot invoke app actions, music controls,
or web tools. Summaries use the local response model. Bodies are limited to
2,000 characters per message; HTML is converted to text without rendering or
loading images. If only a snippet is available, or text is shortened, that
limitation is included in the snapshot. Raw bodies are not saved by the reader.
Summaries remain in the in-memory chat until **New conversation**. After an email
read, that part of the chat and subsequent turns are excluded from web retrieval
history to avoid supplying private email details to search planning. Current
explicit web requests still go online.

The modular interface is `GmailMailReader.read_unread(EmailRequest(limit=5))`
in `actions/gmail_reading.py`. Inject `email_reader=` into `Assistant` to replace
it, or use `email_reader=False` to disable it. `app_control=False` disables the
default reader for embedded clients. Sign-in is a separate explicit UI operation,
never something the LLM or an email can trigger. Dependencies are installed by
`uv pip install --python .venv/bin/python -r actions/requirements-email.txt`.

An optional `OutlookMailReader` remains available for Outlook versions that
expose a real default Inbox through scripting. It uses a fixed getter-only
script and never launches or selects messages. If scripting exposes zero
accounts, it reports **mailbox unavailable**, rather than falsely reporting no
unread email. Native Automation permission is broader than this reader’s exposed
capability; the Gmail API instead enforces a genuinely read-only OAuth scope.

### Web search and page reading

**Settings > Internet > Enable web search and page reading** is enabled by
default. It applies to Text chat, Voice chat and background wake conversations.
Quit and reopen Airis after updating the code. Try:

- **Search the web for Python documentation.**
- **What is the latest news about Apple?**
- **What is the current price of the Mercedes-Benz stock?**
- **Summarize https://docs.python.org/3/library/urllib.parse.html.**

Explicit search requests, a public HTTP/HTTPS link, and questions with cues such
as **latest**, **current**, **today**, **news** or **weather** trigger retrieval.
Ordinary conversation stays local. Clear requests use a fast route; other
information questions, weather requests and short follow-ups use a bounded local
LLM plan to choose local chat, search, news, page reading or one essential
clarification. Stable facts can still be answered locally; ask explicitly when
you want online verification. A request to answer without the internet suppresses
retrieval for that turn.

Source preferences apply across topics, not just stocks. For example, **Explain
black holes, use nasa.gov as source**, **Latest science news from bbc.com**, or
**Explain photosynthesis, use Wikipedia as source** restrict retrieval to that
website and its subdomains. Search results and every page redirect are checked;
another provider is never silently substituted. Website addresses work generally;
common publisher names are recognized immediately. Other names in **use [name]
as source** are looked up through a ranked public homepage search with hostname
matching; an unresolved name requires its website address. The local planner can
also interpret other publisher wording and distinguish **news from Nature**
(a source) from **news from Germany** (a region). A supplied article
URL is read directly, while a homepage
requested as a source is searched for the actual subject. All subjects, including
prices, use this same search and reading pipeline.

Missing optional details use the common reasonable interpretation or recent user
context, with material assumptions stated briefly. Broad questions get a useful
overview rather than a demand for exact identifiers. Essential personal details
are not guessed: **What is the weather today?** asks for a city if none was given;
after **Weather in Berlin**, **And tomorrow?** can reuse Berlin. Follow-up chains
can retain an explicitly requested source; a new topic resets that preference.
Only the standalone search query goes online. Recent user messages used to
resolve context stay with the local model; private plans never enter speech.

Airis searches for up to four general results and attempts to read the first two
pages, trying later results when these pages cannot be read. News requests use
dated Google News feeds, with DDGS news search as a fallback, instead of category
pages. Airis keeps up to six recent article summaries with publisher and
publication date, tries to locate original article links, and prioritizes the
main/article body when reading a page. Undated or stale news and obvious quote
landing pages are excluded. Feed-only links are labeled as summaries, never as
article bodies that were read.
The local model privately checks whether the evidence answers the question. When
an essential fact is missing, it can refine the query or read a discovered detail
link. Up to two additional retrieval steps are allowed, with a time budget checked
between operations; repeated queries and already-read pages are skipped. It stops
early when the evidence is sufficient. Plans stay out of speech and history;
the UI shows concise research progress. The final model receives the collected
evidence and streams an answer. Numbered source links appear below the reply in both
transcripts. Speech replies use natural language without a spoken link list.
News source rows also show the publisher and publication date. Generated source
footers are filtered from streamed output before they reach the UI or speech.
Clicking a source yourself opens it in your browser; the model does not control
the browser. Retrieval failures are reported rather than presented as a live
answer. A blocked page may leave only its search snippet available.

The [DDGS search adapter](https://github.com/deedy5/ddgs) needs no API key. Search
queries go to DuckDuckGo/Brave through DDGS, with Google/Yahoo and then Bing/Mojeek
as bounded fallbacks when earlier providers fail. Source constraints remain in
each fallback query. News retrieval also contacts Google News; the rest of the conversation is not
sent to search providers. Reading a page contacts that website. Qwen, Whisper
and speech synthesis remain local. Retrieved information is context for the
reply, not model training, and no downloaded files are saved. Source identities
and short URLs remain in the session history; long feed URLs stay in the transcript
to avoid filling the model context with opaque identifiers. Disabling web access prevents
new retrieval; it does not remove earlier conversation context.

The page tool performs GET requests only, without cookies, logins, form
submissions or JavaScript. It rejects local/private addresses, credentials and
non-web schemes, rechecks redirects, pins validated public IPs with normal TLS
verification, and limits page sizes and network waiting. Web content is treated
as untrusted evidence and cannot call the app tools. Public HTML, plain text and
JSON are supported. Tables, metadata and embedded JSON are preserved as evidence;
related public links can be selected for reading without executing scripts.
If the standard reader receives HTTP 403, a general browser-compatible HTTP
fallback retains IP pinning and TLS verification, disables proxies, cookies and
automatic redirects, and limits received bytes and time. It applies to any public
website and contains no provider-specific paths or parsers. The fallback uses
[curl_cffi](https://curl-cffi.readthedocs.io/en/stable/advanced.html).
PDF files, data available only after executing JavaScript, paywalls and signed-in
pages are outside this basic version.

There are no dedicated finance adapters or company/provider-specific quote
endpoints. **What is the latest stock price of Amazon, use finanzen.net as source**
uses the same steps as a product price, a timetable or a science question:
search, read, check coverage and follow useful evidence when needed. For numeric
answers the model checks the unit, identity and date/context in the actual
evidence, distinguishes observations from forecasts, and identifies material
assumptions. It must not invent missing values or call dated observations live.
Blocked or dynamically unavailable information can still prevent an exact
answer; the reply states the specific gap instead of silently changing sources.

Install the dependency in the shared environment with:

```bash
uv pip install --python .venv/bin/python -r actions/requirements-web.txt
```

`actions/web.py` exposes replaceable `WebAccess`, `DDGSSearch` and
`PublicPageReader` adapters. Inject `web_access=` into `Assistant`, or use
`web_access=False` to disable the web service for an embedded client. The base
`processing.process_text` function performs no web retrieval. Live checks use
public websites with a simulated response model and no microphone or speech:

```bash
AIRIS_RUN_WEB_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_web_runtime.py' -v
```

To verify the actual 9B news answers silently, first **Quit Airis**, then run:

```bash
AIRIS_RUN_WEB_MODEL_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_web_model_runtime.py' -v
```

The separate model-planning check verifies general recommendations, weather,
missing locations and contextual follow-ups without microphone or speech:

```bash
AIRIS_RUN_WEB_MODEL_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_web_planning_model_runtime.py' -v
```

Do not keep another model session open during this check on a 16 GB Mac. Source
links are explicitly ended in the transcript so later messages stay plain text.

### Background keyboard voice control

While Airis is running, these global shortcuts work without bringing its main
window forward, including when wake-word listening is disabled:

- **Control + A** toggles a background voice session.
  On start, Airis says **I'm ready.** in the selected voice, then waits for your
  talk shortcut. This fixed English cue is cached per voice, uses no LLM and
  creates no conversation-history entry. It plays once per session start.
- **Control + Space** records while held and sends on release.
  After the spoken reply, the session remains ready with the microphone closed.
  Hold again for your next question; you can pause mid-question without sending it.
- Hold the talk shortcut during a reply to request interruption. Recording begins
  after the current inference operation or audio chunk stops, if you are still
  holding the shortcut. Releasing before recording starts sends nothing.

End the session with the A shortcut again or click the circle. Ending while
recording discards the audio. Recordings remain limited to 30 seconds; reaching
that limit sends once and waits for a fresh press. Session history remains shared
with text and voice chat. Closing or focusing another window does not cancel a
background shortcut recording.

**Settings > Background voice** contains the background input mode, an
**Enable global voice shortcuts** switch and registration status. Both settings
persist. A shortcut-started session always uses hold-to-talk; the input selector
controls wake-word-started background sessions. The menu-bar icon also provides
**Start / end voice session** and plays the same readiness cue. Session startup
waits for standby capture to close, prepares the voice runtime if needed, and
pauses media before the cue. Recording cannot start until the cue has finished.
Media stays paused until the session ends.

These shortcuts are easy to remember, but overlap with existing macOS shortcuts.
Control + A normally moves to the beginning of a line or paragraph; it controls
Airis while global voice shortcuts are enabled. Disable Airis's global shortcuts
to restore that text-editing behavior. Control + Space can switch input sources.
If it conflicts, change or disable **Select the previous input source** in
**System Settings > Keyboard > Keyboard Shortcuts > Input Sources**, then click
**Retry shortcuts** in Airis. Airis does not change system keyboard preferences.
See [Apple's keyboard shortcuts](https://support.apple.com/en-us/102650).
Airis registers only these chords using the macOS public
hot-key API, with press/release events rather than general keyboard monitoring.
If registration fails, Settings shows the error and **Retry shortcuts** retries
after the conflicting shortcut is released. Global shortcut registration needs
no Accessibility or Input Monitoring permission; microphone permission is still
required to record. Quitting Airis unregisters both shortcuts. The native adapter
is in `ui/keyboard_shortcuts.py` and can be replaced independently of voice capture.

### Experimental wake-word listening

In **Settings > Wake word**, turn on **Enable wake word (experimental)**. Wait for
**Wake word: Listening**, then say **Hey Airis** or **Airis**. Airis answers with
the fixed cue **I'm listening.** in the selected voice. Wait for that cue to finish.
When the main window is active, speak your question; a pause of about 1.2 seconds
sends the recording automatically. Airis waits up to ten seconds for
your question; silence returns to wake listening without a model request. A
background activation instead starts a continuous voice session. The default
**Settings > Background voice > Input** is **Hold shortcut to talk**: hold
**Control + Space** to record and release it to send. The
microphone is closed between questions, so background sounds cannot create
another turn. Choose **Automatic listening** to use speech and pauses instead,
with no timeout while waiting for the next question. Each question is limited to 30 seconds.
The Talk button becomes **Cancel recording** while waiting for or recording it.

Possible wake matches receive two local checks before activation: a small
Silero speech detector rejects non-speech, then the prepared Whisper confirms a
complete "Airis" or "Hey Airis" utterance. Observed Whisper spellings such as
"Hey Errors" are accepted only with the acoustic match and speech check, never
as name corrections in chat. Ordinary sentences do not pass phrase confirmation.
The classifier candidate threshold is capped at 0.85 in this verified mode;
a high score alone no longer activates the app. A candidate includes 240 ms of
following audio to avoid cutting off the name, and confirmation adds recognition
latency before the cue. The same inference worker serializes verification with
Qwen and TTS; there is no additional Whisper model or idle GPU transcription.

The wake phrase never enters conversation history. Its microphone stream closes
before verification and the cue, and a fresh stream opens afterwards for the
question. Neither accepted nor rejected candidates are saved. The cue calls no
LLM, creates no history entry, and is cached in memory during voice preparation
for immediate reuse. Changing the voice refreshes the cue. Only the question and
its model reply enter conversation history. Speak the question after the cue;
words spoken together with the wake phrase are not buffered into the new question.

The switch is off by default and its selection persists. Enabling it prepares
Whisper, the speech check, and the selected voice, including from Text chat.
The small, pinned Silero v6.2.3 ONNX model is downloaded during preparation if
missing, then reused offline. Its MIT notice is in `input/licenses/SileroVAD_LICENSE`;
include the notice when distributing the model. Library clients can install it
explicitly with `.venv/bin/python -m input.speech_activity --install`.
Wake listening runs on a separate CPU thread while Airis is open, even when
another app has focus.
Detection switches to Voice chat and replies aloud. Typed messages still produce
silent replies. Closing the window keeps an enabled listener available; quitting
Airis stops it. Hold to talk remains available with wake listening on.

When the main window is hidden, minimized, or behind another app, a small animated
circle appears near the lower-right corner of the current display after a wake
activation. It shows the acknowledgement, listening, thinking, and speaking
states without opening the main window or taking keyboard focus. The same voice
pipeline and conversation history continue in the background. After each reply
finishes playing, Airis returns to **Hold to talk** without another wake word or
acknowledgement. In Automatic listening, it opens a fresh question stream instead.
The indicator remains visible through multiple requests and
long pauses. It does not record while Airis is processing or speaking.
Click the circle in any active state to end the session, discard an unfinished
recording, and stop further reply output at the next inference or audio-chunk
boundary. An inference call or audio chunk already in progress finishes before
its resources are released. Cancelled transcription cannot submit a new model
request or app action. The circle disappears when the session ends, wake listening
is disabled, or a capture/model/playback error occurs; its animation stops while
hidden. Right-click it for **Open Airis**, **End voice session**, or **Quit Airis**.
When the main window is active, its existing status replaces the floating circle.
Airis must be running; this does not launch a fully quit app or start it at login.

The visible indicator shows Loading, Listening, Acknowledging, Awaiting question,
Ready to talk, Recording, Processing, or Speaking.
Wake detection pauses during processing, playback, and push-to-talk. Its microphone
stream closes before processing and resumes after the turn, so Airis does not
activate itself while speaking and playback can refresh the system output device.
Question capture uses energy-based silence detection, which can end early or
late in noisy environments, only in Automatic listening. Hold-to-talk sends on
release, including when you pause mid-question. No microphone audio or conversation is saved.

**Show wake-word test controls** displays classifier scores, activation counts,
the active microphone, input level, received audio duration, and buttons to mark
missed wake words and false activations. Peaks include every 80 ms model frame,
even between display updates. **Confirmed peak** is the highest minimum score of
two adjacent frames, matching the two-frame candidate rule. Final activation
also requires speech and phrase confirmation; rejected candidates have a
separate counter. The visible wake-microphone hint remains available without
diagnostics, and **Check wake phrase** uses that same standby microphone and
verification path. These are manual
session counters, not measured recall or a false-activation rate. Try both phrases
at different distances, with headphones and with ordinary background sounds.
The standalone classifier is experimental: its held-out synthetic recall is 38.8%
for Hey Airis and 54.0% for Airis, despite 93.8% combined recall on clean clips.
Real microphone accuracy and battery use have not been measured.

If neither phrase activates, click **Check wake phrase (5 seconds)** and say one
phrase when the status asks you to speak. This check works with wake listening
off as well. It captures the system input device, displays what Whisper heard,
and evaluates that same recording with the wake detector. No question is sent to
Qwen, no spoken reply is generated, and the audio is discarded. An empty input
signal points to microphone selection, muting, or permission. A clear transcript
with a confirmed peak below the activation threshold points to the classifier
or its threshold. A short successful test does not establish accuracy.

For explicit sensitivity experiments, enable **Use a custom detection threshold**.
The initial custom value is 0.9; the calibrated threshold remains the default.
Lower thresholds make activation easier but increase false activations. Compare
your phrase's confirmed peak with the selected threshold and test ordinary
conversation as well. **Reset peak reading** starts a fresh listening window
without reloading the inference models. **Use calibrated threshold** removes the
override. Custom choices persist in Settings and never modify model artifacts,
acceptance metrics, or training data. The listener reports an error if no audio
callbacks arrive instead of remaining at Listening indefinitely.

If a compatible personal evaluation profile is installed, test controls also
offer **Use evaluated headset threshold**. Applying this preset is explicit;
opening Airis never enables it automatically. The preset is tied to the ONNX
model's file hash, so a replacement classifier cannot reuse stale evaluation
settings through that button. Personal recordings are used only for evaluation
and calibration, not classifier training. The currently evaluated 0.85 preset
detected all six supplied EPOS wake recordings and did not activate on one
21.95-second ordinary-speech recording across four timing alignments. This short
development check does not establish a general false-activation rate; the broader
synthetic validation shows frequent false activations at similar low thresholds.

The local inference package is already installed in `input/models/wakeword/`.
Runtime uses only the ONNX classifier, its JSON metadata, and the frozen Google
ONNX frontend and preprocessing constants; no training dependencies or datasets
are loaded. ONNX Runtime is included in `input/requirements.txt`. After retraining,
quit Airis and install the replacement explicitly:

```bash
.venv/bin/python -B -m input.wakeword_assets \
  --source training/wakeword/runs/hey_airis_v2/models/airis_wakeword.onnx
```

Installation verifies artifact hashes and preserves failed acceptance flags.
Missing or invalid artifacts show a retryable error without disabling text chat
or push-to-talk. Runtime files are ignored by Git and must be included separately
when distributing the app. Future product packaging and license review remain
separate from this local experimental integration.

**Reply length** defaults to 2,048 tokens and can be increased to 8,192 in the
interface. It is a ceiling, not a required answer length; completed answers still
stop early. Replies stream into the conversation as they are generated. If a reply
hits the selected limit, the status explains that you can increase it or ask Airis
to continue. Model, reply length, chat area, voice, and language settings persist across
launches. **Model** offers 9B and 4B in 4-bit MLX format; changing it prepares the
selected model while keeping conversation history.

**Voice** selects the built-in **Low**, **Mid**, or **High** reference for spoken
replies; **Mid** is the default. Changing it in Settings after using Voice chat prepares the selected
reference in the background and reuses the loaded TTS model. Wait for **Ready**
before sending the next message. After using Text chat, choosing a voice only saves the
selection and does not load TTS. Voice changes preserve conversation history.

Preparing the model removes first-message loading/compilation delays. It does not
make 9B generate tokens as quickly as a smaller model or eliminate memory pressure.
On Apple silicon, MLX Audio loads the original `openai/whisper-large-v3-turbo`
FP16 weights and converts their layout in memory. Opening **Voice chat** prepares and warms
speech recognition and the streaming TTS path in advance. Text-only startup
does not load Whisper or TTS. In Voice chat, complete sentences from the streamed
LLM response enter a bounded TTS queue. The opening sentence waits for a complete
text sentence boundary before synthesis begins. Playback starts once about one
second of audio is ready, without waiting for the full waveform. A reply shorter
than that plays as soon as generation finishes.
Line breaks and the usual text chunk limit do not split that opening sentence;
an unfinished final sentence is flushed when text generation ends. All sentences
stream through MLX Audio in roughly 0.32-second audio chunks. Text generation
temporarily yields to speech synthesis when the queued audio reserve is low,
then resumes when enough audio is ready or that synthesis finishes. This reduces
GPU contention while preserving the selected model and reference voice. If the
audio device reports an underrun, playback collects a larger reserve before
resuming; the buffer can grow to two seconds and is retained for later turns.
The chunk interval describes audio duration,
not a guaranteed generation latency. A slow model can still cause playback gaps.
The first MLX speech preparation downloads its own converted checkpoint.

Speech timing statistics and device underrun counts appear in `logs/desktop.log`
when launched as an app. These statistics contain no conversation text or audio.
For a silent comparison using full models, quit Airis and run:

```bash
.venv/bin/python -B -m ui.benchmark_speech
```

The benchmark compares concurrent 9B generation with buffered speech priority,
then checks 4B using the same prompt. It loads one response model at a time and
simulates playback at the natural audio rate; it plays and saves no audio.
Results are written to `logs/speech_benchmark.json`. Simulated delivery gaps are
diagnostic timing measurements, not measurements of the actual output device.
The desktop default remains 9B; the benchmark does not change settings.

In Text chat, type a message and press Enter or Send. In Voice chat, hold **Hold to talk**, then
release it to transcribe and send the message. Each recording is limited to
30 seconds. Losing window focus cancels an active push-to-talk recording; an
enabled wake listener remains available in the background. Both input methods
use the same assistant session and conversation history.

While you hold Talk, Airis transcribes recording snapshots in the background
and shows a provisional **Hearing:** preview. It runs at most one recognition
job at a time, without queuing stale snapshots. On release, it reuses the latest
preview only when the audio matches and adds at most half a second of silence;
otherwise it transcribes the complete final recording before sending text to
Qwen. Provisional words never enter conversation history or trigger a response.
Cancelling a recording discards its preview. This uses provisional transcription
of growing recordings, not word-by-word commitments from an incremental decoder.

Qwen keeps a bounded, in-memory checkpoint of the rendered conversation up to
the previous user prompt. The next turn prefills only the new suffix when its
tokens exactly extend that checkpoint. This handles Qwen3.5's recurrent layers
and the template's treatment of thinking tags without trimming recurrent state.
New conversations, changed system instructions, incompatible prompt prefixes,
and interrupted generation invalidate the checkpoint. Retained checkpoints are
limited to 128 MB; no conversation state is saved to disk.

Text chat always produces silent replies; Voice chat always produces spoken replies.
In Settings, **Language** selects
English or German explicitly; **Auto** detects the transcription language and
lets TTS infer the reply language. **New conversation** clears the session
history without reloading the models. All interface text is English.

On macOS, automatic audio selection refreshes the system device defaults before
each spoken reply and each recording, while no Airis audio stream is open. Select
your headphones in **Control Center > Sound** or **System Settings > Sound > Output**;
the next reply uses that output even if the headphones connected after Airis started.
An ongoing reply keeps its output until playback finishes. Explicit `device`
arguments in the playback and microphone APIs still select that device.

On macOS, background wake listening uses the built-in microphone when the system
input is a Bluetooth headset. This keeps Airis from holding the headset in its
lower-quality call profile while music or videos play. A wired system input is
kept when available. Question recording still uses the selected system input,
and that microphone closes before Airis speaks. The wake-word diagnostics show
the actual standby microphone; check detection again after changing microphones.

Starting a voice conversation pauses the current macOS Now Playing item before
the acknowledgement or recording. It stays paused through the question and the
spoken reply, then resumes after playback finishes. In a continuous background
voice session, media stays paused between questions until you end the session.
Cancellation, silence in a single-turn interaction and
errors also release the interruption. Text chat leaves media alone. Only an item
that Airis successfully paused can be resumed; a changed item or one manually
resumed during the conversation is left alone. Other apps are never launched.
Players must expose their playback state through macOS Now Playing. Browser
players may also need **System Settings > Privacy & Security > Accessibility**
permission for Airis (or the terminal when launching from there). If a player
cannot be paused or resumed, Airis displays a notice instead of toggling playback
blindly. Another app using the Bluetooth microphone can still lower headphone
quality even when Airis is idle or closed.

Speech replies append the instructions in `processing/speech_system_prompt.txt`
to the normal system prompt for that turn. They request natural conversational
prose, a short complete opening sentence targeting eight words or fewer, and no headings, Markdown, lists, or
tables. Custom base instructions are retained. Switching back to Text uses the
normal prompt again; conversation history stays shared between both modes.
Spoken replies also avoid correcting the speaker's spelling or pronunciation of
the assistant's name unless explicitly asked.

The interface defaults to `mlx-community/Qwen3.5-9B-4bit` through MLX.
Whisper uses the requested
[`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
checkpoint directly through MLX Audio on Apple silicon, reusing the original
cached weights without downloading a separate converted model.
The explicit `load_whisper_backend` function retains the Transformers backend
for comparison or other platforms. Qwen is prepared at startup; speech mode
also prepares recognition and synthesis. Models are reused between turns;
the first recognition preparation downloads the original Whisper checkpoint if needed.
Model work and playback run outside the UI thread. Microphone recordings and
generated speech remain in memory; the interface saves no audio files.

macOS must allow the launching terminal/Python application to use the microphone.
If permission was denied, enable it under System Settings → Privacy & Security
→ Microphone, then restart the app. A microphone/device error appears in the
status area. Actual microphone capture and full-model inference need a live
check on the target machine.

### Reuse the input modules

`input/microphone.py` records mono audio at the microphone's native sample rate.
`input/transcription.py` validates the recording, resamples it to Whisper's
16 kHz input, and returns source-language text. Silence returns an empty string
without loading Whisper. Keep capture, transcription, response generation, and
playback separate when adding other interfaces or agents:

```python
from input.microphone import MicrophoneRecorder
from input.transcription import transcribe_audio

recorder = MicrophoneRecorder()
recorder.start()  # Call when push-to-talk is pressed.
audio = recorder.stop()  # Call when it is released.
text = transcribe_audio(audio, language="English")
if text:
    reply = assistant.respond(text, mode="text")
```

Install and test the interface/input environment with:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_input.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_live_transcription.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_prompt_cache.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_conversation_workflow.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_app_actions.py' -v
AIRIS_RUN_QT_TESTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -B -m unittest discover -s tests -p 'test_desktop.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_launcher.py' -v
```

Input checks use real Whisper feature extraction with a simulated model. Workflow
checks verify typed/spoken input, shared history, silence handling, and optional
playback without Qt. Desktop checks use headless Qt widgets with simulated models
and microphones; they are opt-in because they initialize the native Qt runtime.
These tests do not access the microphone or download Whisper
weights. Run the desktop checks in a normal local terminal.
Desktop tests also verify background preparation, streamed replies before
generation completes, configurable reply limits, and model reuse. Launcher tests
validate the app bundle and shell entry point without starting a visible window.

For silent checks with the full models, quit Airis and run:

```bash
AIRIS_RUN_VOICE_RUNTIME_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_voice_runtime.py' -v
```

These opt-in checks download Whisper when needed, compare its MLX transcription
with the preserved reference transcript, verify cached and uncached Qwen replies,
and measure a recorded voice turn with simulated playback. They use no microphone
and play or save no audio. `logs/voice_latency.json` holds the timing result.
Real microphone turns log `release_to_first_audio_s`, transcription time, and
time to the first text in `logs/desktop.log`; the first-audio measurement is when
the initial PCM chunk is submitted for playback, before device output latency.

On macOS, startup clears the hidden-file flag on bundled Qt plugin libraries when
necessary. Qt's discovery skips hidden libraries, which can otherwise produce
`Could not find the Qt platform plugin "cocoa"`. The startup check preserves all
other file flags and adds the installed plugin directory to Qt's search paths.

## Assistant sessions: text and optional speech

`assistant.py` connects processing to speech output. The response always contains
text and includes WAV audio when speech mode is requested:

```python
from assistant import Assistant
from processing.mlx_model import load_mlx_backend

# Use the 4B model already verified on the 16 GB Mac.
backend = load_mlx_backend(model_id="mlx-community/Qwen3.5-4B-4bit")
with Assistant(backend=backend) as assistant:
    reply = assistant.respond("Hello, Airis.", mode="text")
    print(reply.text)  # TTS is not started in text mode.

    reply = assistant.respond("Tell me something interesting.", mode="speech")
    print(reply.text)
    reply.audio.save("reply.wav")

    # Switch back to silent text output in the same conversation.
    reply = assistant.respond("Explain that further.", mode="text")
```

The assistant loads no models at construction. Each session owns its conversation
history; switching output modes preserves that history. `history` returns a copy,
and `reset_history()` starts a new conversation. Text mode never calls speech
synthesis. It leaves an already loaded speech worker idle for reuse; `close()`
or exiting the context manager releases a worker owned by the session.

On Apple silicon, speech mode starts a persistent MLX worker using
`.venv/bin/python` and `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`.
The worker reuses the model and the unchanged mid reference voice across turns.
The older PyTorch adapter remains in the source for custom runtimes, but its
`qwen_tts` dependency is excluded from the shared environment because it pins an
incompatible Transformers version. The desktop app uses MLX Audio.
Model loading logs appear in the terminal or `logs/desktop.log` when launched as
an app. Text mode imports no audio/model dependencies and never starts TTS.

For streaming integration, pass `on_audio` to `Assistant.respond` in speech mode:

```python
from assistant import Assistant
from output.playback import AudioStreamPlayer

with Assistant() as assistant, AudioStreamPlayer() as player:
    reply = assistant.respond(
        "Explain how rain forms.", mode="speech", max_new_tokens=512,
        on_text=lambda chunk: print(chunk, end="", flush=True),
        on_audio=player.write,
    )
```

Audio callbacks run sequentially on the playback thread; text callbacks run on
the calling thread. The method waits for playback to finish before returning and
committing history. A streamed response has `audio=None` because chunks are
consumed without retaining a complete recording. Omitting `on_audio` preserves
the complete-WAV API and optional saving. Playback or synthesis errors stop queued
work and propagate to the caller; speech already played cannot be retracted.

`AssistantResponse.audio` contains `wav_bytes` and `sample_rate`. Audio playback
belongs to the UI or calling application. A future microphone adapter can pass
transcribed text to the same `respond` method. A future agent implements the
existing response backend interface. Alternative local or remote speech engines
implement `synthesize(text, *, language) -> SpeechAudio` and can be injected via
`speech_backend`. They can additionally implement `stream(text, *, language)`
to yield incremental `SpeechAudio` chunks. Without that method, Airis synthesizes
one complete sentence at a time. Injected services remain owned by the caller.

If processing or speech synthesis fails, the exception is reported and the
conversation turn is not committed. No audio is played automatically.

Streaming checks without model downloads or audio devices:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_streaming_speech.py' -v
```

An optional full-model check downloads the MLX checkpoint, prepares the mid voice,
and reports first-chunk latency and total generation time without playing or saving:

```bash
AIRIS_RUN_MLX_SPEECH_TESTS=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_mlx_speech_runtime.py' -v
```

### Try the connected flow

Run from the processing environment; the TTS worker selects its own environment:

```bash
source .venv/bin/activate
python -u -m assistant "Introduce yourself in one sentence." --model mlx-community/Qwen3.5-4B-4bit --mode text --max-new-tokens 64
python -u -m assistant "Introduce yourself in one sentence." --model mlx-community/Qwen3.5-4B-4bit --mode speech --language English --max-new-tokens 64 --output reply.wav
```

The speech command saves a WAV file; on macOS, play it with `afplay reply.wav`.
Saving happens only when `--output` is explicitly provided. To play the generated
audio directly from memory, install the optional playback dependency and use
`--play`:

```bash
python -m pip install -r output/requirements-playback.txt
python -u -m assistant "Introduce yourself in one sentence." --model mlx-community/Qwen3.5-4B-4bit --mode speech --language English --max-new-tokens 64 --play
```

The playback module uses [sounddevice raw output streams](https://python-sounddevice.readthedocs.io/en/latest/api/raw-streams.html)
and creates no audio files. In a product, call `output.playback.play_audio(reply.audio)`
or send `reply.audio.wav_bytes` to your UI's audio player. Playback is optional and
does not run in text mode.

The first speech request may download TTS weights if they are not cached.
Keep an `Assistant` instance alive in the product to reuse both models between
requests. The one-shot CLI closes the speech worker when it finishes.

## Processing text with Qwen3.5

Start with the text-to-text entry point in `processing/text_processing.py`:

```python
from processing import process_text

response = process_text("Introduce yourself in one sentence.")
```

Apple Silicon defaults to `mlx-community/Qwen3.5-9B-4bit` with MLX. Other
platforms default to `Qwen/Qwen3.5-9B` with Transformers. The model loads on the
first call and is reused for later requests. `response` is the final answer as a string. Thinking is
disabled in the chat template, and thinking blocks are removed before returning
text. Incomplete thinking or a pending tool call raises an error.
The system instructions are stored in `processing/system_prompt.txt`.

### Local model setup

For a Mac with 16 GB of unified memory, use quantized MLX weights. The unquantized
9B model's 16-bit weights alone require roughly 18 GB, before macOS, application
memory, and generation overhead. Memory pressure and swapping can make this
configuration very slow. The 4-bit model reduces weight memory significantly;
actual latency still depends on context length, output length, and other apps.

Install the Apple Silicon dependencies in the processing environment:

```bash
source .venv/bin/activate
python -m pip install -r processing/requirements-mlx.txt
python -u -c 'from processing import process_text; print(process_text("Introduce yourself in one sentence.", max_new_tokens=64))'
```

The first MLX run downloads a separate quantized checkpoint. To prioritize
latency and leave more memory for speech output, explicitly select the 4B model:

```python
from processing import process_text
from processing.mlx_model import load_mlx_backend

backend = load_mlx_backend(model_id="mlx-community/Qwen3.5-4B-4bit")
response = process_text("Hello, Airis.", backend=backend, max_new_tokens=64)
```

Keep the process running to reuse the loaded model. MLX is designed for Apple
Silicon and supports quantized inference; see the [MLX LM documentation](https://github.com/ml-explore/mlx-lm)
and the [quantized Qwen3.5-9B checkpoint](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit).

### Transformers backend

The shared `.venv` includes Transformers with Qwen3.5 support. The Transformers
backend is available alongside MLX:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r processing/requirements.txt
.venv/bin/python -c 'from processing import process_text; from processing.qwen_model import load_qwen_backend; print(process_text("Hello, Airis.", backend=load_qwen_backend()))'
```

The first model load downloads the weights unless they are already cached.
Model memory requirements depend on dtype and runtime; 9 billion parameters
alone require roughly 18 GB at 16-bit precision, before runtime overhead.
The loader selects CUDA, MPS, or CPU according to availability. CPU defaults to
float32; accelerators default to bfloat16. To configure loading explicitly:

```python
from processing import process_text
from processing.qwen_model import load_qwen_backend

backend = load_qwen_backend(device="mps")
response = process_text("Hello, Airis.", backend=backend, max_new_tokens=256)
```

On MPS, the loader defaults `HF_DEACTIVATE_ASYNC_LOAD` to `1` to load weights
sequentially. This avoids the concurrent Metal weight-copy path seen in native
crash reports; generation still runs on the GPU. An explicit value for this
environment variable takes precedence. If a previous run crashed at
`Loading weights: 0%`, start a new process and retry:

```bash
source .venv/bin/activate
HF_DEACTIVATE_ASYNC_LOAD=1 python -u -c 'from processing import process_text; from processing.qwen_model import load_qwen_backend; print(process_text("Introduce yourself in one sentence.", backend=load_qwen_backend(), max_new_tokens=64))'
```

### Conversation history and future agents

History is owned by the calling application; requests do not share conversation
state. Pass previous user and assistant messages explicitly:

```python
history = []
text = "My name is Alex."
response = process_text(text, history=history)
history.extend([
    {"role": "user", "content": text},
    {"role": "assistant", "content": response},
])
response = process_text("What is my name?", history=history)
```

`process_text` accepts any backend implementing
`respond(messages, *, max_new_tokens) -> str`. A future agent can implement this
interface, manage its own tool execution and memory, and return its final answer
without changing the text input or speech output functions. Tool execution is
not implemented yet.

### Connecting processing to speech output

To use a hosted Qwen model, supply a Chat Completions endpoint.
`ChatAPIBackend` uses Python's standard library and can call that server from
the shared `.venv`:

```python
from processing import process_text
from processing.chat_api import ChatAPIBackend
from output.speech_output import text_to_speech

backend = ChatAPIBackend(base_url="http://localhost:8000/v1")
response = process_text("Introduce yourself.", backend=backend)
audio, sample_rate = text_to_speech(response)
```

This example requires a running Qwen server; Airis does not start one
automatically. The backend requests non-thinking responses using Qwen's
`chat_template_kwargs`. Configure `api_key` and `model_id` if your server requires
them. Hosting options and the model interface are documented in the official
[Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

### Verification

```bash
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_qwen_runtime.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_text_processing.py' -v
.venv/bin/python -B -m unittest discover -s tests -p 'test_mlx_processing.py' -v
```

The runtime smoke test saves and loads a tiny local Qwen3.5 checkpoint and runs
real generation. It does not download the 9B weights or assess answer quality.
Other tests cover conversation isolation, prompt handling, output decoding, API
errors, and the processing-to-speech handoff with a simulated TTS model.
MLX tests simulate the library to verify backend selection, reuse, and generation
arguments without loading Metal or downloading quantized models. They do not
measure performance on Apple Silicon.

## Integrating speech output

The complete-audio product entry point is `output/speech_output.py`. On Apple
silicon its default uses MLX Audio in `.venv` and the existing mid reference voice:

```python
from output.speech_output import text_to_speech

llm_response = "Hello, how can I help you?"  # Response text from your LLM
audio, sample_rate = text_to_speech(llm_response)

# Optionally also save a WAV file:
audio, sample_rate = text_to_speech(llm_response, output_path="reply.wav")
```

The first call loads the voice cloning model and the default voice profile.
Subsequent calls reuse both. The function expects the complete response text;
the application handles audio playback. Use `language` to select the language.
To use a model and voice prompt prepared by your application, pass them together
as `model` and `voice_prompt`.

## Legacy PyTorch model and voice prompt helpers

The examples in this section require the optional `qwen_tts` package and are
retained for custom runtimes. They are excluded from the shared `.venv`; the
desktop app uses the MLX backend and existing reference recordings instead.

Importing the functions does not load a model or create audio files.
Prepare the model and voice prompt once, then reuse them for subsequent
responses:

```python
from output.voice_clone import create_voice_prompt, generate_speech, load_clone_model

model = load_clone_model()
voice_prompt = create_voice_prompt(model)

audio, sample_rate = generate_speech(
    model,
    "Hello, how can I help you?",
    voice_prompt,
)

# Optionally also save a WAV file:
generate_speech(model, "Everything is ready.", voice_prompt, output_path="reply.wav")
```

`audio` contains the response audio data; `sample_rate` is its sample rate.
The integrating application handles playback.
The original texts and voice instructions are preserved in `output/prompts.json`.
Pass any response text directly to `generate_speech` and use `language` to
select its language.

## Generating a reference voice

```python
from output.voice_design import generate_reference, load_design_model
from output import load_prompts

design_model = load_design_model()
reference_text = load_prompts()["voice_design"]["text"]
audio, sample_rate = generate_reference(
    design_model,
    text=reference_text,
    output_path="my_reference.wav",
)

# Use the cloning model loaded above for subsequent responses:
voice_prompt = create_voice_prompt(
    model,
    reference_audio="my_reference.wav",
    reference_text=reference_text,
)
```

The reference text must exactly match the speech in the reference recording.
The default profile uses `output/voice_profiles/mid/reference.wav` and the
original `voice_design.text` transcript, as does the `high` profile. The `low` profile uses its preserved
`voice_clone.reference_text` transcript when selected explicitly. Other reference
recordings require an explicit matching transcript. The voice design text also
belongs to newly generated recordings using the default design text.
`generate_reference` uses the original seed of 7 by default and resets the
global random generators. Pass `seed=None` to skip this step.

## Voice profiles and audio samples

`output/voice_profiles` contains the `low`, `mid`, and `high` voice profiles.
Each profile includes a `reference.wav` for voice cloning and a `samples`
directory containing `introduction.wav` and `morning_briefing.wav` for listening.
The corresponding text keys are documented in
[output/voice_profiles/README.md](output/voice_profiles/README.md).

## Legacy voice generation helpers

`output/voice_design.py` and `output/voice_clone.py` retain the original PyTorch
reference-generation helpers and prompts. Their model loaders require `qwen_tts`,
which is not installed in the shared environment. They are not used by the desktop
app or the default Apple silicon speech API. Existing reference recordings and
transcripts remain available to the MLX speech backend without regeneration.
