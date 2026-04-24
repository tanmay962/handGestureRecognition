import json
import threading
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


WORD_LIST = (
    # Core / function words
    "the,be,to,of,and,a,in,that,have,I,it,for,not,on,with,he,as,you,do,at,this,but,"
    "his,by,from,they,we,her,she,or,an,will,my,one,all,would,there,their,what,so,up,"
    "out,if,about,who,get,which,go,me,when,make,can,like,time,no,just,him,know,take,"
    "people,into,year,your,good,some,could,them,see,other,than,then,now,look,only,"
    "come,its,over,think,also,back,after,use,two,how,our,work,first,well,way,even,"
    "new,want,because,any,these,give,day,most,us,"
    # AAC core
    "great,need,help,home,water,food,right,please,thank,yes,stop,sorry,hello,"
    "name,where,here,open,close,hot,cold,big,small,happy,sad,hungry,thirsty,"
    "tired,sick,pain,doctor,family,friend,love,feel,understand,speak,hear,wait,"
    "eat,drink,sleep,walk,sit,stand,more,less,very,much,many,few,ready,okay,"
    # People / relationships
    "old,young,man,woman,child,boy,girl,mother,father,brother,sister,son,daughter,"
    "husband,wife,baby,nurse,teacher,person,everyone,someone,nobody,together,"
    # Body
    "hand,head,eye,ear,nose,mouth,face,arm,leg,foot,body,heart,chest,stomach,"
    "back,shoulder,knee,throat,tooth,finger,blood,breath,skin,bone,"
    # Places
    "door,window,house,room,school,hospital,clinic,pharmacy,store,market,park,"
    "car,bus,train,road,bathroom,kitchen,bed,chair,table,office,airport,station,"
    "outside,inside,upstairs,downstairs,street,city,"
    # Time
    "today,tomorrow,yesterday,morning,afternoon,evening,night,week,month,hour,"
    "minute,always,never,sometimes,soon,already,again,still,before,after,during,"
    "while,since,until,early,late,noon,quickly,slowly,"
    # Health / medical
    "emergency,ambulance,medicine,medication,allergy,fever,temperature,pressure,"
    "dizzy,nausea,vomit,unconscious,bleeding,broken,swollen,infection,surgery,"
    "prescription,appointment,insurance,wheelchair,oxygen,"
    # Daily / social
    "breakfast,lunch,dinner,soup,bread,milk,coffee,tea,juice,fruit,meal,"
    "phone,message,address,call,visit,meeting,news,money,problem,question,"
    "answer,idea,important,careful,safe,ready,busy,free,quiet,loud,"
    # Connectors / grammar helpers
    "am,is,are,was,were,has,had,been,being,did,does,may,might,shall,should,"
    "must,let,going,trying,trying,having,getting,using,doing,making,saying,"
    "too,both,each,every,same,different,another,last,next,own,few,less,more"
).split(',')

# Frequency weights — boosts common words in suggestion ranking even when
# n-gram context is ambiguous. Values are additive score bonuses (0–0.005).
_FREQ_BONUS = {
    'i':0.005,'you':0.005,'please':0.005,'help':0.005,'need':0.005,
    'want':0.004,'thank':0.004,'yes':0.004,'no':0.004,'hello':0.004,
    'sorry':0.003,'good':0.003,'go':0.003,'stop':0.003,'water':0.004,
    'food':0.003,'the':0.004,'a':0.003,'is':0.003,'are':0.003,
    'to':0.003,'and':0.003,'for':0.003,'not':0.003,'can':0.003,
    'me':0.003,'my':0.003,'here':0.002,'home':0.003,'now':0.003,
    'more':0.002,'am':0.002,'okay':0.002,'ready':0.002,'sick':0.002,
    'pain':0.002,'doctor':0.002,'family':0.002,'hungry':0.002,'tired':0.002,
}


NLP_PHRASES = [
    # Greetings / social
    "hello how are you today",
    "good morning how are you feeling",
    "good afternoon nice to see you",
    "good evening how was your day",
    "my name is nice to meet you",
    "goodbye see you again soon",
    "thank you very much for helping",
    "you are very kind thank you",
    "I am happy to see you",
    "how are you feeling today",
    # Needs / requests
    "please help me I need water",
    "I need food and water please",
    "I am hungry please give me food",
    "I am thirsty I need water please",
    "can I have some water please",
    "I want to eat something please",
    "I need more food thank you",
    "please bring me something to drink",
    "I would like some help please",
    "can you help me please",
    # Medical
    "I feel sick I need a doctor",
    "I need to go to the hospital",
    "I am in a lot of pain",
    "please call the doctor for me",
    "I have a headache please help",
    "I am having trouble breathing",
    "I need my medication right now",
    "please call an ambulance for me",
    "I have an allergy to medicine",
    "my chest hurts I need help",
    "I feel dizzy and need to sit",
    "please get the nurse for me",
    "I need to see the doctor today",
    "I have a high temperature please",
    "I think I need emergency help",
    # Location / navigation
    "I want to go home now",
    "where is the bathroom please",
    "where is the doctor please",
    "can you take me home please",
    "I need to find the bathroom",
    "please open the door for me",
    "I want to go outside please",
    "where is the nearest pharmacy",
    # Understanding / communication
    "I do not understand please repeat",
    "can you speak more slowly please",
    "yes I understand you completely",
    "no thank you sorry about that",
    "please write it down for me",
    "I cannot hear you please speak louder",
    "can you show me what you mean",
    "please say that again more slowly",
    # Emotions / state
    "I am very happy today thank you",
    "I am tired and need to rest",
    "I am scared please stay with me",
    "I am sad please talk to me",
    "I feel much better thank you",
    "I am not feeling well today",
    "I am worried about my family",
    "I am confused please help me",
    # Daily needs
    "please call my family for me",
    "I need to use the bathroom",
    "I want to go to sleep now",
    "I need a blanket I am cold",
    "I am too hot please open window",
    "please turn the light on for me",
    "I need help getting up please",
    "can you sit with me please",
    # Affirmations / responses
    "yes please I would like that",
    "no I do not want that",
    "I am sorry I did not mean",
    "please stop that thank you",
    "okay I understand thank you very much",
    "yes that is right please continue",
    "no that is not right please",
    "I agree with you completely yes",
    # More complex sentences
    "I need to call my family today",
    "please help me find my medicine",
    "can someone help me right now",
    "I feel sick and I need rest",
    "thank you for your help today",
    "I need help right now please",
    "good morning I am feeling well today",
    "please wait I need help right now",
    "can you hear me please respond",
    "I want to go to the store",
    "I need to take my medicine now",
    "please give me some more water",
]


# Language model state

_base_lm        = None
_base_vocab     = set()
_personal_lm    = None
_personal_corpus = []
_gesture_map    = {}
_retrain_lock   = threading.Lock()


def build_base_model():
    global _base_lm, _base_vocab

    tokenized  = [p.lower().split() for p in NLP_PHRASES]
    train_data, vocab = padded_everygram_pipeline(4, tokenized)
    lm = KneserNeyInterpolated(4)
    lm.fit(train_data, vocab)

    _base_lm    = lm
    _base_vocab = set(vocab)
    print(f"[NLP] Kneser-Ney 4-gram model ready ({len(NLP_PHRASES)} seed phrases)")


def build_personal_model():
    global _personal_lm

    if len(_personal_corpus) < 3:
        _personal_lm = None
        return

    def _rebuild():
        global _personal_lm
        try:
            tokenized  = [s.lower().split() for s in _personal_corpus if s.strip()]
            train_data, vocab = padded_everygram_pipeline(4, tokenized)
            lm = KneserNeyInterpolated(4)
            lm.fit(train_data, vocab)
            with _retrain_lock:
                _personal_lm = lm
            print(f"[NLP] Personal 4-gram model retrained on {len(_personal_corpus)} sentences")
        except Exception as e:
            print(f"[NLP] Personal model retrain failed: {e}")

    t = threading.Thread(target=_rebuild, daemon=True)
    t.start()


def load_personal_corpus_from_db():
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


# Suggestion functions

def kn_suggest(words, n=5, use_personal=True):
    ctx        = list(words[-3:]) if len(words) >= 3 else list(words)
    candidates = set(w.lower() for w in WORD_LIST) | (_base_vocab - {'<s>', '</s>', '<UNK>'})
    scores     = {}

    with _retrain_lock:
        personal = _personal_lm

    if use_personal and personal is not None:
        for w in candidates:
            try:
                s = personal.score(w, ctx)
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

    # Add frequency bonus so common words win ties
    for w in candidates:
        bonus = _FREQ_BONUS.get(w, 0)
        if bonus:
            scores[w] = scores.get(w, 0) + bonus

    if not scores:
        return ['I', 'please', 'help', 'thank', 'need']

    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]


def gesture_context_suggest(words, recent_gestures, n=5):
    base = kn_suggest(words, n=n * 3, use_personal=True)

    if not recent_gestures or not _gesture_map:
        return base[:n]

    boost = {}
    for g in recent_gestures:
        for w in _gesture_map.get(g, []):
            boost[w] = boost.get(w, 0) + 1

    if not boost:
        return base[:n]

    scored = {w: boost.get(w, 0) for w in base}
    for w, sc in boost.items():
        if w not in scored:
            scored[w] = sc

    return [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:n]] or base[:n]


def _edit_distance_1(a, b):
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    # Substitution (same length)
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    # Insertion / deletion
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    i = 0
    while i < len(shorter) and longer[i] == shorter[i]:
        i += 1
    return longer[i + 1:] == shorter[i:]


def word_prefix_suggest(prefix, n=5, user_vocab=None):
    p = prefix.lower()
    if not p:
        return []

    vocab = list(set(w.lower() for w in WORD_LIST))
    if user_vocab:
        # user-spelled words always appear first when they prefix-match
        user_matches = sorted(
            [w for w in user_vocab if w.startswith(p) and w != p],
            key=lambda w: -user_vocab[w],
        )
    else:
        user_matches = []

    exact  = [w for w in vocab if w.startswith(p) and w != p and w not in user_matches]

    if len(user_matches) + len(exact) >= n:
        return (user_matches + exact)[:n]

    # Fuzzy: tolerate 1-character error anywhere in the typed prefix
    if len(p) >= 3:
        already = set(user_matches + exact)
        fuzzy = [
            w for w in vocab
            if w not in already and w != p and len(w) > len(p)
            and _edit_distance_1(p, w[:len(p)])
        ]
        exact += fuzzy

    return (user_matches + exact)[:n]


# Grammar correction

_STATE_ADJ = {
    'hungry', 'thirsty', 'tired', 'sick', 'happy', 'sad', 'scared', 'worried',
    'confused', 'ready', 'busy', 'fine', 'okay', 'good', 'bad', 'cold', 'hot',
    'dizzy', 'weak', 'nervous', 'bored', 'excited', 'lost', 'hurt', 'well',
}

_MOTION_VERBS = {'go', 'come', 'walk', 'drive', 'run', 'take', 'bring', 'move'}

_PLACES = {
    'home', 'hospital', 'store', 'school', 'clinic', 'pharmacy', 'park',
    'office', 'bathroom', 'kitchen', 'outside', 'upstairs', 'church', 'market',
    'airport', 'station', 'hotel', 'work',
}

_BARE_INF = {
    'eat', 'drink', 'sleep', 'go', 'come', 'walk', 'sit', 'stand', 'see',
    'hear', 'speak', 'wait', 'open', 'close', 'call', 'help', 'leave', 'stay',
    'take', 'find', 'use', 'rest', 'return', 'stop',
}

_ACTION_VERBS = {
    'want', 'need', 'like', 'love', 'hate', 'go', 'see', 'hear', 'feel',
    'know', 'think', 'eat', 'drink',
}


def offline_grammar_correct(sentence):
    if not sentence or not sentence.strip():
        return None

    words   = sentence.strip().lower().split()
    changed = False
    i       = 0
    out     = []

    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None

        # "I [state-adjective]" → "I am [adj]"  (ASL drops "to be")
        if w == 'i' and nxt in _STATE_ADJ:
            out += ['i', 'am']
            changed = True
            i += 1
            continue

        # "I [noun]" handled by check: "I doctor" → skip (too risky to auto-fix)

        # "[motion-verb] [place]" → "[motion-verb] to [place]"
        if w in _MOTION_VERBS and nxt in _PLACES:
            out.append(w)
            out.append('to')
            changed = True
            i += 1
            continue

        # "want/need [bare-infinitive]" → "want/need to [infinitive]"
        if w in ('want', 'need', 'like', 'try') and nxt in _BARE_INF:
            out.append(w)
            out.append('to')
            changed = True
            i += 1
            continue

        out.append(w)
        i += 1

    words = out

    # "me" at sentence start → "I"
    if words and words[0] == 'me':
        words[0] = 'i'
        changed  = True

    # Implicit subject: action verb at start → prepend "I"
    if words and words[0] in _ACTION_VERBS:
        words.insert(0, 'i')
        changed = True

    # Capitalise all standalone "i"
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

    # Add period if missing
    if words and words[-1][-1] not in '.!?':
        words[-1] += '.'
        changed    = True

    return ' '.join(words) if changed else None


# Sentence state

class SentenceState:
    def __init__(self):
        self.words            = []
        self.spelling         = ''
        self.context          = []
        self.suggestions      = ['I', 'please', 'help', 'thank', 'need']
        self.word_suggestions = []
        self.completion       = None

    def add_word(self, w):
        self.words.append(w.lower())
        self.context.append(w.lower())
        if len(self.context) > 50:
            self.context = self.context[-30:]
        self.spelling         = ''
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
