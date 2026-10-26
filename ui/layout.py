"""Desktop presentation for settings, silent text chat, and voice chat."""

from pathlib import Path
from .keyboard_shortcuts import SESSION_SHORTCUT, TALK_SHORTCUT

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QScrollArea,
    QStackedWidget, QVBoxLayout, QWidget, QPushButton,
)


PAGE_IDS = ("text", "voice", "settings")

STYLE = """
QWidget { color: #e8edf4; font-family: 'Helvetica Neue'; font-size: 14px; }
QWidget#AppBody { background: #10141d; }
QWidget#SettingsBody, QWidget#VoiceBody { background: #10141d; }
QFrame#Sidebar { background: #151b26; border-right: 1px solid #293140; }
QLabel#Brand { font-size: 27px; font-weight: 700; }
QLabel#Kicker { color: #7bdac7; font-size: 11px; font-weight: 600; }
QLabel#PageTitle { font-size: 30px; font-weight: 600; }
QLabel#Subtitle, QLabel#Hint { color: #9ba9bc; }
QLabel#Hint { font-size: 12px; }
QLabel#SectionTitle { font-size: 17px; font-weight: 600; }
QLabel#FieldLabel { font-size: 13px; font-weight: 600; }
QLabel#Badge { color: #7bdac7; background: #1c3938; border-radius: 10px;
              padding: 6px 12px; font-size: 12px; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 14px 16px; margin: 3px 0; border-radius: 9px; color: #9ba9bc; }
QListWidget::item:hover { background: #202b3a; color: #e8edf4; }
QListWidget::item:selected { background: #26413f; color: #93edda; }
QListWidget::item:disabled { color: #66778a; }
QFrame#Card { background: #171e2a; border: 1px solid #293140; border-radius: 14px; }
QTextBrowser { background: #171e2a; color: #e8edf4; border: 1px solid #293140;
               border-radius: 14px; padding: 18px; font-size: 15px; selection-background-color: #315c61; }
QPlainTextEdit { background: #171e2a; color: #e8edf4; border: 1px solid #374355;
                 border-radius: 12px; padding: 12px; font-size: 15px; selection-background-color: #315c61; }
QPlainTextEdit:focus { border-color: #70cdbb; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #202a38; color: #e8edf4; border: 1px solid #374355;
                     border-radius: 8px; padding: 10px 12px; min-height: 20px; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox::down-arrow { image: url('__CHEVRON__'); width: 16px; height: 16px; }
QComboBox QAbstractItemView { background: #202a38; color: #e8edf4;
                             selection-background-color: #315c61; border: 1px solid #374355; }
QPushButton { background: #202a38; color: #dbe4ef; border: 1px solid #374355;
              border-radius: 8px; padding: 10px 16px; }
QPushButton:hover { background: #2b394b; border-color: #576b80; }
QPushButton:pressed { background: #344557; }
QPushButton:focus { border-color: #93edda; }
QPushButton#PrimaryButton { background: #7bdac7; color: #102c28; border: none; font-weight: 600; }
QPushButton#PrimaryButton:hover { background: #a0eddf; }
QPushButton#TalkButton { background: #7bdac7; color: #102c28; border: 8px solid #27423f;
                        border-radius: 90px; font-size: 22px; font-weight: 600; }
QPushButton#TalkButton:hover { background: #a0eddf; border-color: #3b645c; }
QPushButton#TalkButton:pressed { background: #bcf3e7; border-color: #7bdac7; }
QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled {
    background: #1b2330; color: #687a8b; border-color: #293140;
}
QPushButton#TalkButton:disabled { background: #243933; color: #7ca99e; border-color: #23362f; }
QProgressBar { background: #263342; border: none; border-radius: 2px; }
QProgressBar::chunk { background: #7bdac7; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 4px; }
QScrollBar::handle:vertical { background: #3e4e63; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


def label(text, name="Hint"):
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    if name == "Badge":
        widget.setWordWrap(False)
    return widget


def page(title, subtitle):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)
    layout.addWidget(label(title, "PageTitle"))
    layout.addWidget(label(subtitle, "Subtitle"))
    return widget, layout


def card(title):
    widget = QFrame()
    widget.setObjectName("Card")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(22, 20, 22, 22)
    layout.setSpacing(12)
    layout.addWidget(label(title, "SectionTitle"))
    return widget, layout


def field(layout, title, control, hint):
    heading = label(title, "FieldLabel")
    heading.setBuddy(control)
    layout.addWidget(heading)
    layout.addWidget(control)
    layout.addWidget(label(hint))


def build_layout(window, selected_page):
    """Arrange supplied controls without starting models or opening devices."""
    body = QWidget()
    body.setObjectName("AppBody")
    root = QHBoxLayout(body)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    sidebar = QFrame()
    sidebar.setObjectName("Sidebar")
    sidebar.setFixedWidth(200)
    side = QVBoxLayout(sidebar)
    side.setContentsMargins(20, 30, 20, 24)
    side.setSpacing(16)
    side.addWidget(label("Airis", "Brand"))
    side.addWidget(label("YOUR ASSISTANT", "Kicker"))
    side.addSpacing(22)
    window.navigation = QListWidget()
    window.navigation.addItems(["Text chat", "Voice chat", "Settings"])
    window.navigation.setAccessibleName("Main navigation")
    side.addWidget(window.navigation, 1)
    side.addWidget(label("One conversation.\nTwo ways to connect."))
    root.addWidget(sidebar)

    content = QVBoxLayout()
    content.setContentsMargins(28, 26, 28, 22)
    content.setSpacing(22)
    toolbar = QHBoxLayout()
    toolbar.addWidget(label("YOUR SPACE", "Kicker"))
    toolbar.addStretch()
    toolbar.addWidget(window.clear)
    content.addLayout(toolbar)
    window.pages = QStackedWidget()
    content.addWidget(window.pages, 1)
    content.addWidget(window.media_notice)

    text_page, text = page("Text chat", "Write a message. Take your time.")
    text.addWidget(label("Text replies · Silent", "Badge"), 0, Qt.AlignmentFlag.AlignLeft)
    text.addWidget(window.conversation, 1)
    text.addWidget(window.message)
    compose = QHBoxLayout()
    keyboard_hint = label("Enter to send · Shift + Enter for a new line")
    keyboard_hint.setWordWrap(False)
    compose.addWidget(keyboard_hint)
    compose.addStretch()
    window.send.setObjectName("PrimaryButton")
    compose.addWidget(window.send)
    text.addLayout(compose)
    window.pages.addWidget(text_page)

    voice_page, voice = page("Voice chat", "A conversation, out loud.")
    voice_page.setObjectName("VoiceBody")
    voice.addStretch(1)
    window.voice_summary = label("", "Badge")
    window.voice_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
    voice.addWidget(window.voice_summary, 0, Qt.AlignmentFlag.AlignHCenter)
    voice.addSpacing(10)
    window.talk.setObjectName("TalkButton")
    window.talk.setFixedSize(180, 180)
    window.talk.setAccessibleName("Hold to talk; release to send")
    voice.addWidget(window.talk, 0, Qt.AlignmentFlag.AlignHCenter)
    instruction = label("Hold the button while you speak.\nRelease it and Airis will answer aloud.", "Subtitle")
    instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
    voice.addWidget(instruction)
    window.transcription_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.transcription_preview.setMinimumHeight(38)
    voice.addWidget(window.transcription_preview)
    voice.addStretch(1)
    window.transcript_toggle = QPushButton("Show transcript")
    window.transcript_toggle.setCheckable(True)
    voice.addWidget(window.transcript_toggle, 0, Qt.AlignmentFlag.AlignLeft)
    window.voice_conversation.setVisible(False)
    window.voice_conversation.setMaximumHeight(190)
    window.voice_conversation.setMinimumHeight(110)
    voice.addWidget(window.voice_conversation)
    window.transcript_toggle.toggled.connect(window.voice_conversation.setVisible)
    window.transcript_toggle.toggled.connect(
        lambda visible: window.transcript_toggle.setText("Hide transcript" if visible else "Show transcript")
    )
    voice_scroll = QScrollArea()
    voice_scroll.setWidgetResizable(True)
    voice_scroll.setWidget(voice_page)
    window.pages.addWidget(voice_scroll)

    settings_page, settings = page("Settings", "Choose your model, voice, and reply length.")
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    settings_body = QWidget()
    settings_body.setObjectName("SettingsBody")
    controls = QVBoxLayout(settings_body)
    controls.setContentsMargins(0, 0, 8, 0)
    controls.setSpacing(16)
    response_card, response = card("Response")
    response_fields = QHBoxLayout()
    response_fields.setSpacing(20)
    model_field, length_field = QVBoxLayout(), QVBoxLayout()
    field(model_field, "Model", window.model, "9B for detailed replies. 4B for faster conversations.")
    field(length_field, "Reply length", window.reply_limit, "Maximum output tokens. Completed replies still stop early.")
    response_fields.addLayout(model_field, 1)
    response_fields.addLayout(length_field, 1)
    response.addLayout(response_fields)
    controls.addWidget(response_card)
    web_card, web = card("Internet")
    web.addWidget(window.web_enabled)
    web.addWidget(label(
        "Ask to search online, paste a page link, or ask about the latest news or today's information. "
        "Airis shows sources in the transcript. Search requests go online; your models run locally."
    ))
    controls.addWidget(web_card)
    email_card, email = card("Email (read only)")
    email.addWidget(window.email_enabled)
    email_buttons = QHBoxLayout()
    email_buttons.addWidget(window.email_connect)
    email_buttons.addWidget(window.email_disconnect)
    email.addLayout(email_buttons)
    email.addWidget(window.email_connection_status)
    email.addWidget(label(
        "Reads the connected Gmail Inbox, including the same unread emails you see in Outlook. "
        "Messages stay unread. No sending, editing, or deleting. Connect Gmail once using "
        "your Google OAuth Desktop client JSON; see README for setup."
    ))
    controls.addWidget(email_card)
    voice_card, speech = card("Voice & language")
    speech_fields = QHBoxLayout()
    speech_fields.setSpacing(20)
    voice_field, language_field = QVBoxLayout(), QVBoxLayout()
    field(voice_field, "Voice", window.voice, "The reference voice used for spoken replies. Mid is the default.")
    field(language_field, "Language", window.language, "Auto detects your spoken language and lets Airis choose the reply language.")
    speech_fields.addLayout(voice_field, 1)
    speech_fields.addLayout(language_field, 1)
    speech.addLayout(speech_fields)
    controls.addWidget(voice_card)
    background_card, background = card("Background voice")
    field(background, "Input", window.background_input,
          "Hold to talk records only while the shortcut is held. Automatic listening sends speech after a pause.")
    background.addWidget(window.shortcuts_enabled)
    background.addWidget(label(
        f"{SESSION_SHORTCUT}: start or end the session.\n"
        f"Hold {TALK_SHORTCUT} to speak; release to send.\n"
        "Holding the talk shortcut during a reply interrupts it before the next recording."
    ))
    background.addWidget(window.shortcuts_status)
    background.addWidget(label(
        "Control + A replaces its usual text-editing shortcut while Airis shortcuts are enabled. "
        "If Control + Space is assigned to input-source switching, change that shortcut in "
        "System Settings > Keyboard > Keyboard Shortcuts > Input Sources, then retry here."
    ))
    background.addWidget(window.shortcuts_retry)
    controls.addWidget(background_card)
    wake_card, wake = card("Wake word")
    wake.addWidget(window.wake_enabled)
    wake.addWidget(window.wake_microphone_hint)
    wake.addWidget(label(
        'Say "Hey Airis" or "Airis". Wait for "I\'m listening", then ask your question. A pause sends it automatically. '
        "Listening works while Airis is running, including when its window is in the background."
    ))
    wake.addWidget(label(
        "Experimental: recognition is still unreliable with noise and echo. "
        "You can always use Hold to talk. Microphone audio stays in memory."
    ))
    wake.addWidget(window.wake_test)
    wake.addWidget(window.wake_diagnostics)
    wake.addWidget(window.wake_level)
    wake.addWidget(window.wake_custom)
    threshold_actions = QHBoxLayout()
    threshold_actions.addWidget(window.wake_threshold)
    threshold_actions.addWidget(window.wake_restore)
    threshold_actions.addWidget(window.wake_reset)
    wake.addLayout(threshold_actions)
    wake.addWidget(window.wake_check)
    wake.addWidget(window.wake_check_result)
    wake.addWidget(window.wake_profile_button)
    wake.addWidget(window.wake_profile_hint)
    test_actions = QHBoxLayout()
    test_actions.addWidget(window.wake_missed)
    test_actions.addWidget(window.wake_false)
    wake.addLayout(test_actions)
    controls.addWidget(wake_card)
    actions = QHBoxLayout()
    actions.addWidget(window.prepare_button)
    actions.addStretch()
    controls.addStretch()
    scroll.setWidget(settings_body)
    settings.addWidget(scroll, 1)
    settings.addLayout(actions)
    window.pages.addWidget(settings_page)

    content.addWidget(window.wake_status)
    content.addWidget(window.status)
    window.progress.setFixedHeight(3)
    window.progress.setTextVisible(False)
    content.addWidget(window.progress)
    root.addLayout(content, 1)
    window.setCentralWidget(body)
    arrow = (Path(__file__).parent / "assets" / "chevron-down.svg").as_posix()
    window.setStyleSheet(STYLE.replace("__CHEVRON__", arrow))
    index = PAGE_IDS.index(selected_page)
    window.navigation.setCurrentRow(index)
    window.pages.setCurrentIndex(index)
