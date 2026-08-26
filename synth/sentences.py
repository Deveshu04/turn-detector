"""Curated Hinglish sentence bank for turn-detection synthesis.

Every template is a COMPLETE conversational turn (statement or question) in
code-switched Hinglish, provided in two scripts:
  - `deva`:  Devanagari with embedded English words in Latin script -> hi-IN voices
  - `latin`: fully romanized -> en-IN voices
Templates may contain {slot} placeholders expanded by corpus.py.

Incomplete variants are NOT stored here; corpus.py derives them via
(a) audio cuts at TTS word boundaries, (b) trailing conjunctions (CONJ_TAILS),
(c) trailing hesitation fillers (FILLER_TAILS).
"""

SLOTS = {
    "product": [
        {"deva": "mobile cover", "latin": "mobile cover"},
        {"deva": "charger", "latin": "charger"},
        {"deva": "earphones", "latin": "earphones"},
        {"deva": "किताबें", "latin": "kitabein"},
        {"deva": "जूते", "latin": "shoes"},
        {"deva": "कुर्ती", "latin": "kurti"},
        {"deva": "laptop bag", "latin": "laptop bag"},
        {"deva": "water bottle", "latin": "water bottle"},
        {"deva": "साड़ी", "latin": "saree"},
        {"deva": "power bank", "latin": "power bank"},
    ],
    "city": [
        {"deva": "दिल्ली", "latin": "Delhi"},
        {"deva": "मुंबई", "latin": "Mumbai"},
        {"deva": "जयपुर", "latin": "Jaipur"},
        {"deva": "पुणे", "latin": "Pune"},
        {"deva": "लखनऊ", "latin": "Lucknow"},
        {"deva": "इंदौर", "latin": "Indore"},
        {"deva": "बैंगलोर", "latin": "Bangalore"},
        {"deva": "कोलकाता", "latin": "Kolkata"},
    ],
    "when": [
        {"deva": "कल शाम तक", "latin": "kal shaam tak"},
        {"deva": "परसों", "latin": "parso"},
        {"deva": "अगले हफ्ते", "latin": "agle hafte"},
        {"deva": "आज रात तक", "latin": "aaj raat tak"},
        {"deva": "सोमवार को", "latin": "Monday ko"},
        {"deva": "दो दिन में", "latin": "do din mein"},
        {"deva": "weekend पे", "latin": "weekend pe"},
    ],
    "amount": [
        {"deva": "पांच सौ रुपये", "latin": "paanch sau rupaye"},
        {"deva": "दो हज़ार रुपये", "latin": "do hazaar rupaye"},
        {"deva": "तीन सौ पचास रुपये", "latin": "teen sau pachaas rupaye"},
        {"deva": "एक हज़ार रुपये", "latin": "ek hazaar rupaye"},
        {"deva": "साढ़े सात सौ रुपये", "latin": "saadhe saat sau rupaye"},
    ],
    "person": [
        {"deva": "मम्मी", "latin": "mummy"},
        {"deva": "भैया", "latin": "bhaiya"},
        {"deva": "boss", "latin": "boss"},
        {"deva": "दीदी", "latin": "didi"},
        {"deva": "दोस्त", "latin": "dost"},
    ],
    "food": [
        {"deva": "बिरयानी", "latin": "biryani"},
        {"deva": "पनीर रोल", "latin": "paneer roll"},
        {"deva": "चाय", "latin": "chai"},
        {"deva": "मोमोज़", "latin": "momos"},
        {"deva": "पिज़्ज़ा", "latin": "pizza"},
        {"deva": "दाल मखनी", "latin": "dal makhani"},
    ],
    "app": [
        {"deva": "Amazon", "latin": "Amazon"},
        {"deva": "Flipkart", "latin": "Flipkart"},
        {"deva": "Zomato", "latin": "Zomato"},
        {"deva": "Swiggy", "latin": "Swiggy"},
        {"deva": "Paytm", "latin": "Paytm"},
        {"deva": "Myntra", "latin": "Myntra"},
    ],
}

# (deva, latin, domain) — complete turns.
TEMPLATES = [
    # ---------- logistics / ecommerce ----------
    ("मेरा order अभी तक deliver नहीं हुआ है", "Mera order abhi tak deliver nahi hua hai", "logistics"),
    ("Tracking ID मुझे WhatsApp पे भेज दो please", "Tracking ID mujhe WhatsApp pe bhej do please", "logistics"),
    ("क्या इस pincode पे cash on delivery available है", "Kya is pincode pe cash on delivery available hai", "logistics"),
    ("मुझे अपना parcel {when} चाहिए", "Mujhe apna parcel {when} chahiye", "logistics"),
    ("Courier वाले ने बोला address गलत है", "Courier wale ne bola address galat hai", "logistics"),
    ("मैंने {app} से {product} order किया था", "Maine {app} se {product} order kiya tha", "logistics"),
    ("Return pickup अभी तक schedule नहीं हुआ", "Return pickup abhi tak schedule nahi hua", "logistics"),
    ("Delivery boy का number मुझे भेज दीजिए", "Delivery boy ka number mujhe bhej dijiye", "logistics"),
    ("मेरा refund कब तक process होगा", "Mera refund kab tak process hoga", "logistics"),
    ("Package {city} से निकल चुका है क्या", "Package {city} se nikal chuka hai kya", "logistics"),
    ("मुझे shipment का status check करना है", "Mujhe shipment ka status check karna hai", "logistics"),
    ("इस order में {product} missing था box में", "Is order mein {product} missing tha box mein", "logistics"),
    ("Exchange करना है मुझे ये {product} size issue की वजह से", "Exchange karna hai mujhe ye {product} size issue ki wajah se", "logistics"),
    ("आपका warehouse {city} में कहां पे है", "Aapka warehouse {city} mein kahan pe hai", "logistics"),
    ("Delivery charge कितना लगेगा {city} तक", "Delivery charge kitna lagega {city} tak", "logistics"),
    ("मेरा order out for delivery दिखा रहा है सुबह से", "Mera order out for delivery dikha raha hai subah se", "logistics"),
    ("गलत product आ गया है मुझे replacement चाहिए", "Galat product aa gaya hai mujhe replacement chahiye", "logistics"),
    ("Invoice की copy email कर दो मुझे", "Invoice ki copy email kar do mujhe", "logistics"),
    ("Order cancel करके पैसे wapas कर दो", "Order cancel karke paise wapas kar do", "logistics"),
    ("पचास orders daily ship करने पे rate क्या मिलेगा", "Pachaas orders daily ship karne pe rate kya milega", "logistics"),

    # ---------- customer support ----------
    ("मेरा complaint number नोट कर लीजिए", "Mera complaint number note kar lijiye", "support"),
    ("कल से आपकी app काम नहीं कर रही", "Kal se aapki app kaam nahi kar rahi", "support"),
    ("मुझे senior से बात करनी है अभी", "Mujhe senior se baat karni hai abhi", "support"),
    ("आपने जो promise किया था वो पूरा नहीं हुआ", "Aapne jo promise kiya tha wo poora nahi hua", "support"),
    ("मेरा account block हो गया है बिना reason के", "Mera account block ho gaya hai bina reason ke", "support"),
    ("OTP नहीं आ रहा मेरे number पे", "OTP nahi aa raha mere number pe", "support"),
    ("ये issue तीसरी बार हो रहा है इस महीने", "Ye issue teesri baar ho raha hai is mahine", "support"),
    ("Customer care का number busy जा रहा है", "Customer care ka number busy ja raha hai", "support"),
    ("मुझे email पे confirmation चाहिए इसका", "Mujhe email pe confirmation chahiye iska", "support"),
    ("ठीक है मैं wait कर लूंगा {when}", "Theek hai main wait kar lunga {when}", "support"),
    ("आप मेरी details verify कर लीजिए पहले", "Aap meri details verify kar lijiye pehle", "support"),
    ("मुझे इस service का subscription cancel करना है", "Mujhe is service ka subscription cancel karna hai", "support"),

    # ---------- payments / banking ----------
    ("मेरे account से {amount} कट गए", "Mere account se {amount} kat gaye", "payments"),
    ("Payment fail हो गया लेकिन पैसे deduct हो गए", "Payment fail ho gaya lekin paise deduct ho gaye", "payments"),
    ("UPI से transfer किया था मैंने {amount}", "UPI se transfer kiya tha maine {amount}", "payments"),
    ("मेरा card खो गया है उसे block कर दो", "Mera card kho gaya hai use block kar do", "payments"),
    ("EMI की date change हो सकती है क्या", "EMI ki date change ho sakti hai kya", "payments"),
    ("Cashback अभी तक credit नहीं हुआ wallet में", "Cashback abhi tak credit nahi hua wallet mein", "payments"),
    ("मुझे statement चाहिए पिछले तीन महीने की", "Mujhe statement chahiye pichhle teen mahine ki", "payments"),
    ("Recharge हो गया but balance नहीं दिखा रहा", "Recharge ho gaya but balance nahi dikha raha", "payments"),
    ("{app} पे {amount} का cashback मिला मुझे", "{app} pe {amount} ka cashback mila mujhe", "payments"),

    # ---------- food ordering ----------
    ("एक {food} order कर दो मेरे लिए", "Ek {food} order kar do mere liye", "food"),
    ("खाना ठंडा आया था इस बार", "Khana thanda aaya tha is baar", "food"),
    ("{food} में extra spicy बोल देना", "{food} mein extra spicy bol dena", "food"),
    ("Restaurant ने order accept नहीं किया अभी तक", "Restaurant ne order accept nahi kiya abhi tak", "food"),
    ("आज dinner में {food} बना लेते हैं", "Aaj dinner mein {food} bana lete hain", "food"),
    ("मुझे भूख लगी है कुछ order करते हैं", "Mujhe bhookh lagi hai kuch order karte hain", "food"),
    ("{app} पे fifty percent off चल रहा है आज", "{app} pe fifty percent off chal raha hai aaj", "food"),
    ("Delivery में एक घंटा दिखा रहा है cancel कर दो", "Delivery mein ek ghanta dikha raha hai cancel kar do", "food"),

    # ---------- travel / commute ----------
    ("मुझे {city} की train ticket book करनी है", "Mujhe {city} ki train ticket book karni hai", "travel"),
    ("Cab वाला location पे नहीं पहुंचा अभी तक", "Cab wala location pe nahi pahucha abhi tak", "travel"),
    ("Flight delay हो गई है दो घंटे", "Flight delay ho gayi hai do ghante", "travel"),
    ("Station से pickup कर लोगे मुझे", "Station se pickup kar loge mujhe", "travel"),
    ("{city} में traffic बहुत ज्यादा है आज", "{city} mein traffic bahut zyada hai aaj", "travel"),
    ("Metro से चलते हैं जल्दी पहुंच जाएंगे", "Metro se chalte hain jaldi pahunch jayenge", "travel"),
    ("Hotel की booking confirm हो गई क्या", "Hotel ki booking confirm ho gayi kya", "travel"),
    ("वापसी की ticket {when} की करा लेना", "Wapsi ki ticket {when} ki kara lena", "travel"),

    # ---------- work / office ----------
    ("Meeting {when} reschedule कर दो please", "Meeting {when} reschedule kar do please", "work"),
    ("मैंने report भेज दी है {person} को", "Maine report bhej di hai {person} ko", "work"),
    ("आज मैं work from home कर रहा हूं", "Aaj main work from home kar raha hoon", "work"),
    ("Client का call आया था दोपहर में", "Client ka call aaya tha dopahar mein", "work"),
    ("Presentation कल सुबह तक ready हो जाएगी", "Presentation kal subah tak ready ho jayegi", "work"),
    ("मेरी leave approve हो गई finally", "Meri leave approve ho gayi finally", "work"),
    ("Deadline {when} तक extend करवा दो", "Deadline {when} tak extend karwa do", "work"),
    ("Laptop की screen फिर से hang हो रही है", "Laptop ki screen phir se hang ho rahi hai", "work"),
    ("Office की cab miss हो गई आज मेरी", "Office ki cab miss ho gayi aaj meri", "work"),
    ("Salary credit हो गई क्या तुम्हारी", "Salary credit ho gayi kya tumhari", "work"),

    # ---------- daily life ----------
    ("{person} को बोल देना मैं late आऊंगा", "{person} ko bol dena main late aaunga", "daily"),
    ("बिजली चली गई है एक घंटे से", "Bijli chali gayi hai ek ghante se", "daily"),
    ("मैं अभी निकल रहा हूं घर से", "Main abhi nikal raha hoon ghar se", "daily"),
    ("दूध खत्म हो गया है लाना पड़ेगा", "Doodh khatam ho gaya hai laana padega", "daily"),
    ("कल बारिश होगी शायद umbrella रख लेना", "Kal baarish hogi shayad umbrella rakh lena", "daily"),
    ("मेरा phone charge नहीं हो रहा ठीक से", "Mera phone charge nahi ho raha theek se", "daily"),
    ("{person} का birthday है अगले हफ्ते gift लेना है", "{person} ka birthday hai agle hafte gift lena hai", "daily"),
    ("घर पहुंच के call करना मुझे", "Ghar pahunch ke call karna mujhe", "daily"),
    ("AC की servicing करवानी है गर्मी से पहले", "AC ki servicing karwani hai garmi se pehle", "daily"),
    ("मैंने नया phone लिया है {app} से", "Maine naya phone liya hai {app} se", "daily"),
    ("पानी की motor चालू कर देना ऊपर से", "Paani ki motor chaalu kar dena upar se", "daily"),
    ("मुझे नींद आ रही है कल बात करते हैं", "Mujhe neend aa rahi hai kal baat karte hain", "daily"),

    # ---------- small talk ----------
    ("और बताओ क्या चल रहा है आजकल", "Aur batao kya chal raha hai aajkal", "smalltalk"),
    ("मैं बिल्कुल ठीक हूं तुम सुनाओ", "Main bilkul theek hoon tum sunao", "smalltalk"),
    ("कल match देखा तुमने रात वाला", "Kal match dekha tumne raat wala", "smalltalk"),
    ("बहुत दिन हो गए मिले हुए यार", "Bahut din ho gaye mile hue yaar", "smalltalk"),
    ("चलो weekend पे plan बनाते हैं कहीं का", "Chalo weekend pe plan banate hain kahin ka", "smalltalk"),
    ("अच्छा सुनो एक काम था तुमसे", "Accha suno ek kaam tha tumse", "smalltalk"),
    ("हां हां बिल्कुल टाइम पे आ जाऊंगा", "Haan haan bilkul time pe aa jaunga", "smalltalk"),
    ("ठीक है फिर बाद में बात करते हैं", "Theek hai phir baad mein baat karte hain", "smalltalk"),
    ("अरे वो movie देखी क्या नई वाली", "Are wo movie dekhi kya nayi wali", "smalltalk"),
    ("मौसम कितना अच्छा हो गया है ना आज", "Mausam kitna accha ho gaya hai na aaj", "smalltalk"),
]

# Trailing conjunctions -> utterance is clearly UNFINISHED (label: incomplete).
CONJ_TAILS = [
    {"deva": "और", "latin": "aur"},
    {"deva": "लेकिन", "latin": "lekin"},
    {"deva": "क्योंकि", "latin": "kyunki"},
    {"deva": "तो", "latin": "toh"},
    {"deva": "मतलब", "latin": "matlab"},
    {"deva": "पर", "latin": "par"},
    {"deva": "फिर", "latin": "phir"},
    {"deva": "अगर", "latin": "agar"},
]

# Trailing hesitation fillers -> speaker is thinking, NOT done (label: incomplete).
FILLER_TAILS = [
    {"deva": "umm", "latin": "umm"},
    {"deva": "मतलब", "latin": "matlab"},
    {"deva": "वो क्या है ना", "latin": "wo kya hai na"},
    {"deva": "हां तो", "latin": "haan toh"},
    {"deva": "actually", "latin": "actually"},
    {"deva": "basically", "latin": "basically"},
    {"deva": "ऐसा है कि", "latin": "aisa hai ki"},
    {"deva": "क्या बोलते हैं उसको", "latin": "kya bolte hain usko"},
]

# Mid-utterance fillers, inserted between words of complete sentences to make
# filler-robust COMPLETE examples (speaker used a filler but still finished).
MID_FILLERS = [
    {"deva": "मतलब", "latin": "matlab"},
    {"deva": "अच्छा", "latin": "accha"},
    {"deva": "हां", "latin": "haan"},
    {"deva": "वो", "latin": "wo"},
    {"deva": "umm", "latin": "umm"},
    {"deva": "यार", "latin": "yaar"},
    {"deva": "बस", "latin": "bas"},
]
