"""
nlp.py — Language model for next-word suggestions and sentence tracking

Uses NLTK's Kneser-Ney trigram LM as the base model, plus a personal
model that retrains automatically from the user's own signing history.
Sentence state is tracked here too since it's closely tied to NLP.
"""

import json
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


# Common words the suggestion system pulls from.
# This is just a starting vocabulary — the KN model scores over it.
WORD_LIST = (
    "the,be,to,of,and,a,in,that,have,I,it,for,not,on,with,he,as,you,do,at,this,but,"
    "his,by,from,they,we,her,she,or,an,will,my,one,all,would,there,their,what,so,up,"
    "out,if,about,who,get,which,go,me,when,make,can,like,time,no,just,him,know,take,"
    "people,into,year,your,good,some,could,them,see,other,than,then,now,look,only,"
    "come,its,over,think,also,back,after,use,two,how,our,work,first,well,way,even,"
    "new,want,because,any,these,give,day,most,us,great,need,help,home,water,food,"
    "right,please,thank,yes,stop,go,sorry,hello,name,where,here,open,close,hot,cold,"
    "big,small,happy,sad,hungry,thirsty,tired,sick,pain,doctor,family,friend,love,"
    "feel,understand,speak,hear,see,wait,eat,drink,sleep,walk,sit,stand,more,less,"
    "very,much,many,few,old,young,man,woman,child,boy,girl,mother,father,brother,"
    "sister,son,daughter,husband,wife,baby,hand,head,eye,ear,nose,mouth,face,arm,"
    "leg,foot,body,heart,back,door,window,house,room,school,work,car,bus,train,road,"
    "today,tomorrow,yesterday,morning,afternoon,evening,night,always,never,sometimes,"
    "soon,already,again,still,before,after,during,while,because,since,until,safe,ready"
).split(',')

# Seed phrases for the base language model.
NLP_PHRASES = [
    "hello how are you today",
    "thank you very much please",
    "please help me I need water",
    "yes I understand you completely",
    "no thank you sorry about that",
    "I am hungry please help me",
    "good morning how are you feeling",
    "please wait I need help now",
    "I want to go home now",
    "stop please listen to me carefully",
    "I feel sick I need doctor",
    "open the door for me please",
    "I am very happy today thank",
    "can you hear me please help",
    "I need food and water please",
    "where is the bathroom please help",
    "I do not understand please repeat",
    "can you speak more slowly please",
    "I am tired and need rest",
    "please call my family for me",
    "I need to go to hospital",
    "thank you for your help today",
    "yes please I would like that",
    "no I do not want that",
    "I am sorry I did not",
    "please tell me where to go",
    "I feel better thank you very",
    "can I have some water please",
    "I need help right now please",
    "good morning I am feeling well",
]


# ── Language model state ──────────────────────────────────────────────────────

_base_lm      = None
_base_vocab   = set()
_personal_lm  = None
_personal_corpus = []      # sentences the user has spoken (learned over time)
_gesture_map  = {}         # gesture name → list of words it tended to produce


def build_base_model():
    """Train the Kneser-Ney trigram LM on seed phrases."""
    global _base_lm, _base_vocab

    tokenized  = [p.lower().split() for p in NLP_PHRASES]
    train_data, vocab = padded_everygram_pipeline(3, tokenized)
    lm = KneserNeyInterpolated(3)
    lm.fit(train_data, vocab)

    _base_lm    = lm
    _base_vocab = set(vocab)
    print(f"[NLP] Kneser-Ney trigram model ready ({len(NLP_PHRASES)} seed phrases)")


def build_personal_model():
    """Retrain personal model from accumulated corpus. Called in a background thread."""
    global _personal_lm

    if len(_personal_corpus) < 3:
        _personal_lm = None
        return

    try:
        tokenized  = [s.lower().split() for s in _personal_corpus if s.strip()]
        train_data, vocab = padded_everygram_pipeline(3, tokenized)
        lm = KneserNeyInterpolated(3)
        lm.fit(train_data, vocab)
        _personal_lm = lm
        print(f"[NLP] Personal model retrained on {len(_personal_corpus)} sentences")
    except Exception as e:
        print(f"[NLP] Personal model retrain failed: {e}")
        _personal_lm = None


def load_personal_corpus_from_db():
    """Load the personal corpus from the DB on startup."""
    from database import get_db

    global _personal_corpus, _gesture_map

    try:
        with get_db() as db:
            rows = db.execute(
                "SELECT sentence, gesture_sequence FROM personal_corpus ORDER BY created_at"
            ).fetchall()

        for row in rows:
            sentence = row['sentence']
            gestures = json.loads(row['gesture_sequence'] or '[]')
            _personal_corpus.append(sentence)
            words = sentence.lower().split()
            for i, g in enumerate(gestures):
                if g and i < len(words):
                    _gesture_map.setdefault(g, []).append(words[i])

        if _personal_corpus:
            print(f"[NLP] Loaded {len(_personal_corpus)} personal sentences from DB")
            build_personal_model()

    except Exception as e:
        print(f"[NLP] Personal corpus load skipped: {e}")


# ── Suggestion functions ──────────────────────────────────────────────────────

def kn_suggest(words, n=5, use_personal=True):
    """Suggest next words using the personal model then the base model."""
    ctx        = list(words[-2:]) if len(words) >= 2 else list(words[-1:]) if words else []
    candidates = set(WORD_LIST) | (_base_vocab - {'<s>', '</s>', '<UNK>'})
    scores     = {}

    # Personal model gets double weight when available
    if use_personal and _personal_lm is not None:
        for w in candidates:
            try:
                s = _personal_lm.score(w, ctx)
                if s > 0:
                    scores[w] = scores.get(w, 0) + s * 2.0
            except Exception:
                pass

    if _base_lm is not None:
        for w in candidates:
            try:
                s = _base_lm.score(w, ctx)
                if s > 0:
                    scores[w] = scores.get(w, 0) + s
            except Exception:
                pass

    if not scores:
        return ['hello', 'I', 'please', 'help', 'thank']

    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]


def gesture_context_suggest(words, recent_gestures, n=5):
    """Boost words that tend to follow the given recent gestures."""
    base = kn_suggest(words, n=n * 3, use_personal=True)

    if not recent_gestures or not _gesture_map:
        return base[:n]

    boost = {}
    for g in recent_gestures:
        for w in _gesture_map.get(g, []):
            boost[w] = boost.get(w, 0) + 1

    if not boost:
        return base[:n]

    scored = [(w, boost.get(w, 0)) for w in base]
    for w, sc in sorted(boost.items(), key=lambda x: -x[1]):
        if w not in [x[0] for x in scored]:
            scored.append((w, sc))

    scored.sort(key=lambda x: -x[1])
    result = [w for w, _ in scored[:n]]
    return result if result else base[:n]


def word_prefix_suggest(prefix, n=5):
    """Return words that start with the given spelling prefix."""
    p = prefix.lower()
    return [w for w in WORD_LIST if w.startswith(p) and w != p][:n]


def offline_grammar_correct(sentence):
    """
    Basic rule-based grammar correction — no external API needed.
    Returns the corrected string, or None if nothing changed.
    """
    if not sentence or not sentence.strip():
        return None

    words   = sentence.strip().lower().split()
    changed = False

    ACTION_VERBS = {'want', 'need', 'like', 'love', 'hate', 'go', 'see',
                    'hear', 'feel', 'know', 'think', 'eat', 'drink'}

    # Add missing "I" before action verbs at the start
    if words and words[0] in ACTION_VERBS:
        words.insert(0, 'i')
        changed = True

    # "me" at the start of a sentence → "I"
    if words and words[0] == 'me':
        words[0] = 'i'
        changed  = True

    # Capitalise inline "i"
    words = ['I' if w == 'i' else w for w in words]

    # Capitalise first word
    if words:
        words[0] = words[0].capitalize()

    # Remove consecutive duplicate words
    deduped = [words[0]] if words else []
    for i in range(1, len(words)):
        if words[i].lower() != words[i - 1].lower():
            deduped.append(words[i])
        else:
            changed = True
    words = deduped

    # Add a period if the sentence doesn't end with punctuation
    if words and words[-1][-1] not in '.!?':
        words[-1] += '.'
        changed    = True

    return ' '.join(words) if changed else None


# ── Sentence state ────────────────────────────────────────────────────────────

class SentenceState:
    """Tracks the current sentence being built by the user."""

    def __init__(self):
        self.words           = []
        self.spelling        = ''
        self.context         = []
        self.suggestions     = ['hello', 'I', 'please', 'help', 'thank']
        self.word_suggestions = []
        self.completion      = None

    def add_word(self, w):
        self.words.append(w.lower())
        self.context.append(w.lower())
        if len(self.context) > 50:
            self.context = self.context[-30:]
        self.spelling       = ''
        self.word_suggestions = []

    def to_dict(self):
        display = ' '.join(self.words)
        if self.spelling:
            display += (' ' if self.words else '') + self.spelling + '_'
        return {
            "words":           self.words,
            "spelling":        self.spelling,
            "context":         self.context,
            "suggestions":     self.suggestions,
            "wordSuggestions": self.word_suggestions,
            "completion":      self.completion,
            "sentence":        ' '.join(self.words),
            "displayText":     display,
        }
