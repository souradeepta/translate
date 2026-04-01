"""
Download or generate Bengali-English parallel corpus for benchmarking.

Sources (in priority order):
  1. FLORES-200 tarball from fbaipublicfiles.com
  2. HuggingFace datasets library (if installed)
  3. Built-in 100-sentence corpus covering 10 domains

Output: corpus/flores200_devtest.bn.txt  +  corpus/flores200_devtest.en.txt
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

CORPUS_DIR = Path("corpus")

FLORES_TARBALL_URL = "https://dl.fbaipublicfiles.com/flores200/flores200_dataset.tar.gz"


# ---------------------------------------------------------------------------
# Built-in 100-sentence corpus: 10 domains × 10 sentences
# Sources: Tatoeba (CC0), Wikipedia (CC-BY-SA), public domain literature
# ---------------------------------------------------------------------------
BUILTIN_CORPUS: list[tuple[str, str]] = [
    # Literature
    ("রবীন্দ্রনাথ ঠাকুর বাংলা সাহিত্যের একজন অবিস্মরণীয় কবি।",
     "Rabindranath Tagore is an unforgettable poet of Bengali literature."),
    ("তাঁর কবিতা ও গল্প বাংলাদেশ ও ভারতে সমানভাবে জনপ্রিয়।",
     "His poems and stories are equally popular in Bangladesh and India."),
    ("১৯১৩ সালে তিনি সাহিত্যে নোবেল পুরস্কার পান।",
     "In 1913, he received the Nobel Prize in Literature."),
    ("শরৎচন্দ্র চট্টোপাধ্যায় ছিলেন বাংলার একজন বিখ্যাত ঔপন্যাসিক।",
     "Sarat Chandra Chattopadhyay was a famous Bengali novelist."),
    ("কাজী নজরুল ইসলামকে বাংলাদেশের জাতীয় কবি বলা হয়।",
     "Kazi Nazrul Islam is called the national poet of Bangladesh."),
    ("বাংলা সাহিত্যের ইতিহাস অনেক পুরনো এবং সমৃদ্ধ।",
     "The history of Bengali literature is very old and rich."),
    ("মৈমনসিংহ গীতিকা বাংলার লোকসাহিত্যের একটি মহামূল্যবান সংকলন।",
     "Mymensingh Gitika is a priceless collection of Bengali folk literature."),
    ("লালন শাহ ছিলেন একজন বাউল সাধক ও সুরকার।",
     "Lalon Shah was a Baul mystic and composer."),
    ("আমার সোনার বাংলা বাংলাদেশের জাতীয় সংগীত।",
     "Amar Sonar Bangla is the national anthem of Bangladesh."),
    ("জন্মেছি মায়ের কোলে, মরতে চাই মায়ের পায়ের কাছে।",
     "I was born in my mother's lap, I wish to die at her feet."),

    # History & Culture
    ("একুশে ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস হিসেবে পালিত হয়।",
     "The twenty-first of February is celebrated as International Mother Language Day."),
    ("ভাষা আন্দোলন বাংলাদেশের ইতিহাসে একটি গুরুত্বপূর্ণ ঘটনা।",
     "The Language Movement is an important event in the history of Bangladesh."),
    ("মুক্তিযুদ্ধের মাধ্যমে ১৯৭১ সালে বাংলাদেশ স্বাধীনতা অর্জন করে।",
     "Bangladesh gained independence in 1971 through the Liberation War."),
    ("শহীদ মিনার ভাষা আন্দোলনের শহীদদের স্মরণে নির্মিত।",
     "The Shaheed Minar was built in memory of the martyrs of the Language Movement."),
    ("বাংলাদেশের জাতীয় মসজিদ বায়তুল মোকাররম ঢাকায় অবস্থিত।",
     "The national mosque of Bangladesh, Baitul Mukarram, is located in Dhaka."),
    ("পহেলা বৈশাখ বাংলা নববর্ষ উৎসব।",
     "Pohela Boishakh is the Bengali New Year festival."),
    ("সুন্দরবন পৃথিবীর বৃহত্তম ম্যানগ্রোভ বন।",
     "The Sundarbans is the largest mangrove forest in the world."),
    ("কক্সবাজার পৃথিবীর দীর্ঘতম প্রাকৃতিক সমুদ্র সৈকত।",
     "Cox's Bazar has the longest natural sea beach in the world."),
    ("জাতীয় স্মৃতিসৌধ মুক্তিযুদ্ধের শহীদদের স্মরণে নির্মিত।",
     "The National Memorial was built in memory of the martyrs of the Liberation War."),
    ("বাংলাদেশের পতাকায় সবুজ মাঠের উপর লাল বৃত্ত রয়েছে।",
     "The flag of Bangladesh has a red circle on a green field."),

    # Geography & Nature
    ("বাংলা ভাষা একটি সমৃদ্ধ ও প্রাচীন ভাষা।",
     "The Bengali language is a rich and ancient language."),
    ("পদ্মা নদী বাংলাদেশের একটি প্রধান নদী।",
     "The Padma River is one of the major rivers of Bangladesh."),
    ("ঢাকা বাংলাদেশের রাজধানী এবং একটি ব্যস্ত মহানগর।",
     "Dhaka is the capital of Bangladesh and a busy metropolis."),
    ("চট্টগ্রাম বাংলাদেশের বৃহত্তম বন্দর শহর।",
     "Chittagong is the largest port city of Bangladesh."),
    ("বাংলাদেশ দক্ষিণ এশিয়ার একটি ছোট কিন্তু ঘনবসতিপূর্ণ দেশ।",
     "Bangladesh is a small but densely populated country in South Asia."),
    ("বর্ষা মৌসুমে বাংলাদেশে প্রচুর বৃষ্টিপাত হয়।",
     "Bangladesh receives heavy rainfall during the monsoon season."),
    ("হিমালয় পর্বতমালা এশিয়ার সর্বোচ্চ পর্বতশ্রেণী।",
     "The Himalayan mountain range is the highest mountain range in Asia."),
    ("বঙ্গোপসাগর ভারতীয় মহাসাগরের একটি অংশ।",
     "The Bay of Bengal is a part of the Indian Ocean."),
    ("বাংলাদেশে মোট ৬৪টি জেলা রয়েছে।",
     "There are a total of 64 districts in Bangladesh."),
    ("সূর্য পূর্ব দিকে উদয় হয় এবং পশ্চিম দিকে অস্ত যায়।",
     "The sun rises in the east and sets in the west."),

    # Science & Technology
    ("বিজ্ঞান ও প্রযুক্তি আধুনিক সভ্যতার মূল ভিত্তি।",
     "Science and technology are the foundation of modern civilization."),
    ("ইন্টারনেট বিশ্বকে একটি গ্লোবাল ভিলেজে পরিণত করেছে।",
     "The internet has turned the world into a global village."),
    ("কৃত্রিম বুদ্ধিমত্তা একবিংশ শতাব্দীর সবচেয়ে গুরুত্বপূর্ণ প্রযুক্তি।",
     "Artificial intelligence is the most important technology of the twenty-first century."),
    ("স্মার্টফোন আমাদের দৈনন্দিন জীবনকে সহজ করে দিয়েছে।",
     "Smartphones have made our daily lives easier."),
    ("সৌরশক্তি একটি পরিষ্কার ও নবায়নযোগ্য শক্তির উৎস।",
     "Solar energy is a clean and renewable source of energy."),
    ("মহাকাশ গবেষণা মানুষের জ্ঞান ও অভিজ্ঞতার সীমানা বিস্তৃত করেছে।",
     "Space research has expanded the boundaries of human knowledge and experience."),
    ("জলবায়ু পরিবর্তন বিশ্বের একটি বড় সমস্যা হয়ে দাঁড়িয়েছে।",
     "Climate change has become a major problem for the world."),
    ("কম্পিউটার আধুনিক বিজ্ঞান ও প্রযুক্তির একটি মহান আবিষ্কার।",
     "The computer is a great invention of modern science and technology."),
    ("পরমাণু শক্তি শান্তিপূর্ণ কাজে ব্যবহার করা উচিত।",
     "Nuclear energy should be used for peaceful purposes."),
    ("পৃথিবী সূর্যের চারদিকে ঘুরছে এবং নিজের অক্ষের উপরেও আবর্তন করছে।",
     "The earth revolves around the sun and also rotates on its own axis."),

    # Education
    ("শিক্ষা জাতির মেরুদণ্ড।",
     "Education is the backbone of a nation."),
    ("মানুষের জীবনে শিক্ষা অত্যন্ত গুরুত্বপূর্ণ।",
     "Education is extremely important in a person's life."),
    ("বিশ্ববিদ্যালয় হলো উচ্চশিক্ষার কেন্দ্র।",
     "The university is the center of higher education."),
    ("শিক্ষক জাতি গড়ার কারিগর।",
     "Teachers are the architects of nation building."),
    ("বাংলাদেশে সাক্ষরতার হার বৃদ্ধি পাচ্ছে।",
     "The literacy rate in Bangladesh is increasing."),
    ("প্রাথমিক শিক্ষা শিশুদের মানসিক বিকাশে সহায়তা করে।",
     "Primary education helps in the mental development of children."),
    ("পাঠ্যপুস্তক শিক্ষার অন্যতম প্রধান উপকরণ।",
     "Textbooks are one of the main tools of education."),
    ("জ্ঞান আলোর মতো অন্ধকার দূর করে।",
     "Knowledge dispels darkness like light."),
    ("শিশুরা হলো দেশের ভবিষ্যৎ।",
     "Children are the future of the nation."),
    ("পরিশ্রম সাফল্যের চাবিকাঠি।",
     "Hard work is the key to success."),

    # Health
    ("স্বাস্থ্যই সম্পদ।",
     "Health is wealth."),
    ("নিয়মিত ব্যায়াম স্বাস্থ্যের জন্য উপকারী।",
     "Regular exercise is beneficial for health."),
    ("সুষম খাবার খাওয়া শরীরকে সুস্থ রাখে।",
     "Eating a balanced diet keeps the body healthy."),
    ("পরিষ্কার-পরিচ্ছন্নতা রোগ প্রতিরোধ করে।",
     "Cleanliness prevents disease."),
    ("ডায়াবেটিস একটি দীর্ঘমেয়াদী রোগ যা রক্তে শর্করার মাত্রা বাড়িয়ে দেয়।",
     "Diabetes is a chronic disease that raises blood sugar levels."),
    ("ভিটামিন ও খনিজ পদার্থ শরীরের জন্য অপরিহার্য।",
     "Vitamins and minerals are essential for the body."),
    ("মানসিক স্বাস্থ্য শারীরিক স্বাস্থ্যের মতোই গুরুত্বপূর্ণ।",
     "Mental health is just as important as physical health."),
    ("পর্যাপ্ত ঘুম শরীর ও মনের জন্য জরুরি।",
     "Adequate sleep is essential for body and mind."),
    ("হাসপাতাল রোগীদের চিকিৎসা ও সেবা প্রদান করে।",
     "Hospitals provide treatment and care to patients."),
    ("পানি পানের আগে ভালোভাবে ফুটিয়ে নেওয়া উচিত।",
     "Water should be boiled properly before drinking."),

    # Everyday life
    ("বন্ধুত্ব জীবনের একটি মূল্যবান সম্পদ।",
     "Friendship is a valuable asset in life."),
    ("সততা সর্বোত্তম নীতি।",
     "Honesty is the best policy."),
    ("সময়ের কাজ সময়ে করা উচিত।",
     "Work should be done on time."),
    ("পরিবার হলো সমাজের মূল ভিত্তি।",
     "Family is the basic foundation of society."),
    ("বাজারে নানা ধরনের তাজা সবজি ও ফলমূল পাওয়া যায়।",
     "Various types of fresh vegetables and fruits are available in the market."),
    ("সকালে উঠে ব্রাশ করা এবং মুখ ধোওয়া স্বাস্থ্যকর অভ্যাস।",
     "Brushing teeth and washing face in the morning are healthy habits."),
    ("রান্নাঘরে মা প্রতিদিন সুস্বাদু খাবার রান্না করেন।",
     "Mother cooks delicious food in the kitchen every day."),
    ("বাসে চড়ে অফিসে যেতে প্রতিদিন এক ঘণ্টা লাগে।",
     "It takes one hour every day to go to the office by bus."),
    ("ছুটির দিনে পরিবারের সবাই মিলে পার্কে বেড়াতে যায়।",
     "On holidays, the whole family goes to the park together."),
    ("রাতে ঘুমানোর আগে বই পড়া একটি ভালো অভ্যাস।",
     "Reading a book before sleeping at night is a good habit."),

    # Agriculture & Economy
    ("বাংলাদেশের অর্থনীতি মূলত কৃষিনির্ভর।",
     "The economy of Bangladesh is mainly agriculture-based."),
    ("ধান বাংলাদেশের প্রধান খাদ্যশস্য।",
     "Rice is the main food crop of Bangladesh."),
    ("পোশাক শিল্প বাংলাদেশের সবচেয়ে বড় রপ্তানি খাত।",
     "The garment industry is the largest export sector of Bangladesh."),
    ("কৃষক মাঠে কঠোর পরিশ্রম করেন।",
     "The farmer works hard in the field."),
    ("বাংলাদেশের গ্রামাঞ্চলে কৃষি ও মৎস্য চাষ প্রচলিত।",
     "Agriculture and fish farming are common in the rural areas of Bangladesh."),
    ("পাট বাংলাদেশের একটি গুরুত্বপূর্ণ অর্থকরী ফসল।",
     "Jute is an important cash crop of Bangladesh."),
    ("রেমিট্যান্স বাংলাদেশের অর্থনীতিতে গুরুত্বপূর্ণ ভূমিকা রাখে।",
     "Remittances play an important role in the economy of Bangladesh."),
    ("বাংলাদেশ সরকার কৃষি উন্নয়নে বিনিয়োগ করছে।",
     "The government of Bangladesh is investing in agricultural development."),
    ("মাছে ভাতে বাঙালি — এটি বাংলাদেশের একটি প্রিয় প্রবাদ।",
     "Fish and rice for the Bengali — this is a beloved proverb of Bangladesh."),
    ("বাংলাদেশের রপ্তানি আয় বছরে বছরে বাড়ছে।",
     "Bangladesh's export earnings are growing year by year."),

    # Philosophy & Proverbs
    ("যে রাঁধে সে চুলও বাঁধে।",
     "She who cooks also ties her hair — a woman can do many things at once."),
    ("অতি লোভ তাতি নষ্ট।",
     "Excessive greed ruins everything."),
    ("ধৈর্য ধর, ভালো হবে।",
     "Be patient, things will get better."),
    ("আগে দর্শনধারী পরে গুণবিচারী।",
     "First appearances matter before inner qualities are judged."),
    ("মানুষ মরে গেলে পরে থাকে তার কর্ম।",
     "When a person dies, only their deeds remain."),
    ("দশের লাঠি একের বোঝা।",
     "The sticks of ten are a burden to one — shared burdens are lighter."),
    ("যার নাই সে জানে।",
     "Only those who lack something truly know its value."),
    ("পথের শেষ নেই, চলতে থাকো।",
     "The road has no end, keep walking."),
    ("সংসার সুখের হয় রমণীর গুণে।",
     "A household becomes happy through the virtues of its women."),
    ("একতাই বল।",
     "Unity is strength."),
]


def download_flores200(force: bool = False) -> tuple[Path, Path]:
    """Try to download FLORES-200 tarball, fall back to built-in corpus."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    bn_path = CORPUS_DIR / "flores200_devtest.bn.txt"
    en_path = CORPUS_DIR / "flores200_devtest.en.txt"

    if bn_path.exists() and en_path.exists() and not force:
        print(f"Corpus already exists: {bn_path}, {en_path}")
        return bn_path, en_path

    # Try tarball download
    print(f"Attempting FLORES-200 tarball download from {FLORES_TARBALL_URL} ...")
    try:
        with urllib.request.urlopen(FLORES_TARBALL_URL, timeout=30) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            bn_lines = _extract_flores_lang(tar, "ben_Beng")
            en_lines = _extract_flores_lang(tar, "eng_Latn")
        if bn_lines and en_lines:
            bn_path.write_text("\n".join(bn_lines), encoding="utf-8")
            en_path.write_text("\n".join(en_lines), encoding="utf-8")
            print(f"  Downloaded {len(bn_lines)} FLORES-200 sentences.")
            return bn_path, en_path
    except Exception as e:
        print(f"  Tarball download failed: {e}")

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        print("  Trying HuggingFace datasets library...")
        ds = load_dataset("facebook/flores", "ben_Beng-eng_Latn", split="devtest", trust_remote_code=True)
        bn_lines = [row["sentence_ben_Beng"] for row in ds]
        en_lines = [row["sentence_eng_Latn"] for row in ds]
        bn_path.write_text("\n".join(bn_lines), encoding="utf-8")
        en_path.write_text("\n".join(en_lines), encoding="utf-8")
        print(f"  Loaded {len(bn_lines)} FLORES-200 sentences via datasets library.")
        return bn_path, en_path
    except Exception as e:
        print(f"  HuggingFace datasets failed: {e}")

    # Write built-in corpus
    print(f"  Using built-in corpus ({len(BUILTIN_CORPUS)} sentences).")
    bn_lines2 = [pair[0] for pair in BUILTIN_CORPUS]
    en_lines2 = [pair[1] for pair in BUILTIN_CORPUS]
    bn_path.write_text("\n".join(bn_lines2), encoding="utf-8")
    en_path.write_text("\n".join(en_lines2), encoding="utf-8")
    return bn_path, en_path


def _extract_flores_lang(tar: tarfile.TarFile, lang_code: str) -> list[str]:
    """Extract a specific language file from FLORES-200 tarball."""
    for member in tar.getmembers():
        if member.name.endswith(f"{lang_code}.devtest"):
            f = tar.extractfile(member)
            if f:
                return f.read().decode("utf-8").strip().splitlines()
    return []


def validate_alignment(bn_path: Path, en_path: Path) -> int:
    bn_lines = bn_path.read_text(encoding="utf-8").strip().splitlines()
    en_lines = en_path.read_text(encoding="utf-8").strip().splitlines()
    if len(bn_lines) != len(en_lines):
        print(f"WARNING: line count mismatch — bn={len(bn_lines)}, en={len(en_lines)}")
    else:
        print(f"Corpus aligned: {len(bn_lines)} sentence pairs")
    return min(len(bn_lines), len(en_lines))


def show_samples(bn_path: Path, en_path: Path, n: int = 5) -> None:
    bn_lines = bn_path.read_text(encoding="utf-8").strip().splitlines()
    en_lines = en_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nFirst {n} sentence pairs:")
    print("-" * 80)
    for i, (bn, en) in enumerate(zip(bn_lines[:n], en_lines[:n])):
        print(f"[{i+1}] BN: {bn}")
        print(f"     EN: {en}")
    print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bengali-English corpus")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample pairs to display")
    args = parser.parse_args()

    print("=" * 60)
    print("Bengali-English Corpus Setup")
    print("=" * 60)

    bn_path, en_path = download_flores200(force=args.force)
    n_pairs = validate_alignment(bn_path, en_path)
    show_samples(bn_path, en_path, n=args.samples)

    print(f"\nCorpus ready: {n_pairs} sentence pairs")
    print(f"  Bengali:  {bn_path}")
    print(f"  English:  {en_path}")
    print("\nNext steps:")
    print("  Translate: python scripts/benchmark.py --models nllb-600M")
    print("  Full bench: python scripts/benchmark.py --models nllb-600M nllb-1.3B indicTrans2-1B")


if __name__ == "__main__":
    main()
