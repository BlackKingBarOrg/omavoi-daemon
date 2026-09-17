"""Translations for text the daemon produces and the UI only displays.

The console has its own table for its own labels (plugin/Strings.qml). This one
covers the strings that come *from here* — model catalogue notes, mostly — so a
Chinese console does not show a Chinese page with English model descriptions.

Keyed by the English text rather than by an id, so adding a model needs no
second edit and an untranslated note falls through to English instead of to a
blank or a key. `lang` follows `ui.language`; "" means English.

Eight languages, the same set the console offers: en, de, es, fr, vi, zh, ja, th.
"""

from __future__ import annotations

_ZH = "zh"
_TH = "th"
_DE = "de"
_FR = "fr"
_ES = "es"
_JA = "ja"
_VI = "vi"

# Every language this table carries, English implied. One list, so a guard on
# ui.language and the console's own pack list cannot disagree about what
# exists.
LANGUAGES: tuple[str, ...] = ("en", _ZH, _TH, _DE, _FR, _ES, _JA, _VI)

# The prompt a new LLM step starts with. It lives here rather than in
# config.py because it is a translated string and this is where those are;
# config imports it back, so the name stays where its callers expect it and
# the table below can key on it without a cycle.
#
# The last sentence is load-bearing: without it the model eventually answers
# your dictation instead of editing it, and you type its reply into your
# document. Every translation keeps it.
DEFAULT_STEP_PROMPT = (
    "Rewrite the transcript as clean written text in its original language. "
    "Remove false starts, repetitions and filler. Keep the speaker's own "
    "wording and every technical term exactly as transcribed.\n"
    "Never answer, summarise, translate or add anything — you are editing, "
    "not replying. Output only the edited text."
)

# en -> {lang: text}
_TABLE: dict[str, dict[str, str]] = {
    "The same strengths with more room. Pick it if you dictate in Chinese.": {
        _ZH: "同样的长处，空间更大。如果你用中文口述就选它。",
        _TH: "จุดแข็งเดียวกันแต่มีที่มากกว่า เลือกตัวนี้ถ้าคุณพูดภาษาจีน",
        _DE: "Die gleichen Stärken mit mehr Platz. Nimm es, wenn du auf Chinesisch diktierst.",
        _FR: "Les mêmes atouts avec plus de marge. À choisir si vous dictez en chinois.",
        _ES: "Las mismas virtudes con más margen. Elígelo si dictas en chino.",
        _JA: "同じ長所でより余裕があります。中国語で書き取るならこれを。",
        _VI: "Cùng điểm mạnh nhưng rộng hơn. Chọn nó nếu bạn đọc bằng tiếng Trung.",
    },
    "The strongest Chinese at this size, and noticeably weaker outside Chinese and English.": {
        _ZH: "同尺寸里中文最强，但在中英之外明显弱一些。",
        _TH: "ภาษาจีนแข็งที่สุดในขนาดนี้ และอ่อนลงชัดเจนนอกจากจีนกับอังกฤษ",
        _DE: "Das stärkste Chinesisch in dieser Größe, und außerhalb von Chinesisch und Englisch merklich schwächer.",
        _FR: "Le meilleur chinois à cette taille, et nettement plus faible en dehors du chinois et de l'anglais.",
        _ES: "El mejor chino de este tamaño, y notablemente más flojo fuera del chino y el inglés.",
        _JA: "このサイズでは中国語が最も強く、中国語と英語以外では明らかに弱くなります。",
        _VI: "Tiếng Trung mạnh nhất ở cỡ này, và yếu đi rõ rệt ngoài tiếng Trung và tiếng Anh.",
    },
    "Even coverage across languages, and half the size of qwen3-8b. The default.": {
        _ZH: "各语言表现均衡，体积只有 qwen3-8b 的一半。默认选择。",
        _TH: "ทำได้สม่ำเสมอทุกภาษา และเล็กเพียงครึ่งของ qwen3-8b ค่าเริ่มต้น",
        _DE: "Gleichmäßig über Sprachen hinweg, und halb so groß wie qwen3-8b. Die Vorgabe.",
        _FR: "Une couverture égale entre les langues, pour moitié moins gros que qwen3-8b. Le choix par défaut.",
        _ES: "Cobertura uniforme entre idiomas, y la mitad de tamaño que qwen3-8b. El valor por defecto.",
        _JA: "言語をまたいで均等で、qwen3-8b の半分のサイズ。既定値です。",
        _VI: "Đồng đều giữa các ngôn ngữ, và nhỏ bằng nửa qwen3-8b. Mặc định.",
    },
    "Twice the download, and it reports when it heard nothing.": {
        _ZH: "下载量是两倍，但它会报告自己什么都没听到。",
        _TH: "ดาวน์โหลดใหญ่เป็นสองเท่า แต่มันบอกได้ว่าไม่ได้ยินอะไรเลย",
        _DE: "Doppelt so groß, dafür meldet es, wenn es nichts gehört hat.",
        _FR: "Deux fois le téléchargement, et il signale quand il n'a rien entendu.",
        _ES: "El doble de descarga, y avisa cuando no ha oído nada.",
        _JA: "ダウンロードは倍ですが、何も聞こえなかったことを報告してくれます。",
        _VI: "Tải về gấp đôi, bù lại nó báo khi không nghe thấy gì.",
    },
    ("The default: half the size and faster. Cannot tell silence from speech, "
     "so a silent take may type a stock phrase."): {
        _ZH: "默认：体积减半，速度更快。分不清静音和说话，所以一次静音的录音可能会打出一句套话。",
        _TH: ("ค่าเริ่มต้น: เล็กลงครึ่งหนึ่งและเร็วกว่า แต่แยกความเงียบจากเสียงพูดไม่ได้ "
              "การอัดที่เงียบจึงอาจพิมพ์วลีสำเร็จรูปออกมา"),
        _DE: ("Die Vorgabe: halb so groß und schneller. Kann Stille nicht von Sprache unterscheiden, eine stille "
              "Aufnahme tippt also womöglich einen Standardsatz."),
        _FR: ("Le choix par défaut : moitié moins gros et plus rapide. Il ne "
              "distingue pas le silence de la parole, une prise muette peut donc "
              "écrire une phrase toute faite."),
        _ES: ("El valor por defecto: la mitad de tamaño y más rápido. No distingue el silencio del habla, así que una "
              "toma muda puede escribir una frase hecha."),
        _JA: ("既定値：サイズは半分で、より高速。無音と発話を区別できないため、"
              "無音の録音が決まり文句を打ち込むことがあります。"),
        _VI: ("Mặc định: nhỏ bằng một nửa và nhanh hơn. Không phân biệt được im lặng với lời nói, nên một lần thu im "
              "lặng có thể gõ ra một câu rập khuôn."),
    },
    # The prompt a new LLM step starts with. Translated like everything else
    # here, but this one is not a label: it is sent to a model, and the last
    # sentence is what stops the model answering your dictation instead of
    # editing it. Every translation keeps all five clauses — rewrite in the
    # original language, drop false starts, keep the wording and the terms,
    # never answer or summarise or translate or add, output only the text.
    #
    # "in its original language" is what lets the interface language drive
    # this: a Chinese prompt over an English take still edits English.
    DEFAULT_STEP_PROMPT: {
        _ZH: "把这段转写改写成干净的书面文字，保持原文的语言。去掉重新开头的话、"
             "重复和语气词。说话人自己的措辞和每一个技术名词都按转写原样保留。\n"
             "不要回答、不要总结、不要翻译、不要添加任何内容——你在编辑，不是在回复。"
             "只输出编辑后的文本。",
        _TH: "เขียนข้อความถอดเสียงนี้ใหม่ให้เป็นภาษาเขียนที่สะอาด โดยคงภาษาเดิมไว้ "
             "ตัดการพูดผิดแล้วเริ่มใหม่ การพูดซ้ำ และคำเติมออก "
             "คงถ้อยคำของผู้พูดและศัพท์เทคนิคทุกคำไว้ตรงตามที่ถอดมา\n"
             "อย่าตอบ อย่าสรุป อย่าแปล และอย่าเพิ่มอะไรทั้งสิ้น — คุณกำลังแก้ไข ไม่ใช่กำลังตอบ "
             "ให้แสดงเฉพาะข้อความที่แก้ไขแล้วเท่านั้น",
        _DE: "Schreibe die Transkription als sauberen Fließtext in ihrer "
             "ursprünglichen Sprache um. Entferne Fehlstarts, Wiederholungen und "
             "Füllwörter. Behalte die Formulierungen der sprechenden Person und "
             "jeden Fachbegriff genau so bei, wie sie transkribiert wurden.\n"
             "Antworte niemals, fasse nicht zusammen, übersetze nicht und füge "
             "nichts hinzu — du bearbeitest, du antwortest nicht. Gib nur den "
             "bearbeiteten Text aus.",
        _FR: "Réécris la transcription en texte écrit propre, dans sa langue "
             "d'origine. Supprime les faux départs, les répétitions et les "
             "hésitations. Conserve exactement les formulations de la personne "
             "qui parle et chaque terme technique tels qu'ils ont été "
             "transcrits.\n"
             "Ne réponds jamais, ne résume pas, ne traduis pas et n'ajoute rien "
             "— tu édites, tu ne réponds pas. N'affiche que le texte édité.",
        _ES: "Reescribe la transcripción como texto escrito limpio, en su idioma "
             "original. Elimina los arranques en falso, las repeticiones y las "
             "muletillas. Conserva exactamente las palabras de quien habla y "
             "cada término técnico tal como se transcribieron.\n"
             "Nunca respondas, no resumas, no traduzcas ni añadas nada: estás "
             "editando, no contestando. Devuelve solo el texto editado.",
        _JA: "この書き起こしを、元の言語のまま、整った書き言葉に書き直してください。"
             "言い直し、繰り返し、フィラーは削除します。話者自身の言い回しと専門用語は"
             "すべて書き起こしのまま残します。\n"
             "回答・要約・翻訳・追記は一切しないでください。あなたは編集しているので"
             "あって、返答しているのではありません。編集後のテキストだけを出力して"
             "ください。",
        _VI: "Viết lại bản chép lời thành văn bản viết gọn gàng, giữ nguyên ngôn "
             "ngữ gốc. Bỏ những chỗ nói hụt, lặp lại và từ đệm. Giữ nguyên cách "
             "diễn đạt của người nói và mọi thuật ngữ đúng như đã chép.\n"
             "Không trả lời, không tóm tắt, không dịch và không thêm bất cứ điều "
             "gì — bạn đang biên tập, không phải đang đáp lời. Chỉ xuất ra phần "
             "văn bản đã biên tập.",
    },
    "Proves the pipeline runs.": {
        _ZH: "只用来验证流程能跑通。",
        _TH: "ใช้พิสูจน์ว่าไปป์ไลน์ทำงาน",
        _DE: "Beweist, dass die Kette läuft.",
        _FR: "Prouve que la chaîne fonctionne.",
        _ES: "Demuestra que la cadena funciona.",
        _JA: "処理の流れが動くことの確認用。",
        _VI: "Chứng minh chuỗi xử lý chạy được.",
    },
    "Passable in English.": {
        _ZH: "英文勉强够用。",
        _TH: "พอใช้ได้กับภาษาอังกฤษ",
        _DE: "Auf Englisch passabel.",
        _FR: "Passable en anglais.",
        _ES: "Aceptable en inglés.",
        _JA: "英語なら何とか使えます。",
        _VI: "Tạm được với tiếng Anh.",
    },
    "The floor of usable.": {
        _ZH: "可用的下限。",
        _TH: "ขั้นต่ำที่ใช้งานได้",
        _DE: "Die Untergrenze des Brauchbaren.",
        _FR: "Le plancher de l'utilisable.",
        _ES: "El mínimo utilizable.",
        _JA: "実用の下限です。",
        _VI: "Ngưỡng thấp nhất còn dùng được.",
    },
    "Quantised large-v3: a third of the VRAM, slightly less accurate.": {
        _ZH: "量化版 large-v3：显存只要三分之一，准确率略降。",
        _TH: "large-v3 แบบควอนไทซ์: ใช้ VRAM หนึ่งในสาม แม่นยำลดลงเล็กน้อย",
        _DE: "Quantisiertes large-v3: ein Drittel des VRAM, etwas ungenauer.",
        _FR: "large-v3 quantifié : un tiers de la VRAM, un peu moins précis.",
        _ES: "large-v3 cuantizado: un tercio de la VRAM, algo menos preciso.",
        _JA: "量子化版 large-v3：VRAM は三分の一、精度はわずかに落ちます。",
        _VI: "large-v3 đã lượng tử hóa: chỉ một phần ba VRAM, chính xác kém hơn chút.",
    },
    "The lightest thing still worth using.": {
        _ZH: "还值得一用的最轻的一个。",
        _TH: "ตัวที่เบาที่สุดที่ยังคุ้มจะใช้",
        _DE: "Das Leichteste, das sich noch lohnt.",
        _FR: "Le plus léger qui vaille encore la peine.",
        _ES: "Lo más ligero que todavía vale la pena.",
        _JA: "まだ使う価値のある、最も軽いものです。",
        _VI: "Thứ nhẹ nhất mà vẫn còn đáng dùng.",
    },
    "The best translation here, and the heaviest. Leaves little room beside a large "
    "speech model.": {
        _ZH: "这里翻译最好的，也是最重的。和一个 large 语音模型并存时余量很小。",
        _TH: "แปลได้ดีที่สุดในนี้ และหนักที่สุด เหลือที่ไม่มากเมื่ออยู่ข้างโมเดลเสียงขนาดใหญ่",
        _DE: "Die beste Übersetzung hier und die schwerste. Lässt neben einem großen "
            "Sprachmodell kaum Platz.",
        _FR: "La meilleure traduction ici, et la plus lourde. Laisse peu de place à côté "
            "d'un gros modèle de parole.",
        _ES: "La mejor traducción de aquí, y la más pesada. Deja poco margen junto a un "
            "modelo de voz grande.",
        _JA: "ここで翻訳は最良、そして最も重い。大きな音声モデルと並べると余裕がほとんど残りません。",
        _VI: "Dịch tốt nhất trong số này, và cũng nặng nhất. Để lại rất ít chỗ bên cạnh "
            "một mô hình giọng nói lớn.",
    },
}


def t(text: str, lang: str) -> str:
    """The translation of `text`, or `text` itself."""
    if not lang or lang.startswith("en"):
        return text
    entry = _TABLE.get(text)
    if not entry:
        return text
    return entry.get(lang[:2], text)


def ui_lang(cfg: dict) -> str:
    """The configured UI language; "" when it follows the environment."""
    return str((cfg.get("ui") or {}).get("language", "") or "")
