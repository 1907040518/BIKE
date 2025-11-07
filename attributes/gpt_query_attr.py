import os
import json
import time
import re
from pathlib import Path
from typing import Dict, List

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain

langchain.debug = True

# ===== 配置 LLM（沿用原有的 Qwen + Ollama 调用方式） =====
API_SECRET_KEY = "ENTER YOUR KEY"
BASE_URL = "https://flag.smarttrot.com/v1/"
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

model = Ollama(
    model="llama3.1:8b",
    temperature=0.7,
    base_url="http://localhost:11434",
)

# ===== 数据集元信息 =====
# DATASET_NAME = "HMDB51"
DATASET_NAME = "UCF101"
# DATASET_NAME = "Kinetics400"
# DATASET_NAME = "Something-Something V2"
WORDS_PER_CLASS = 32
REQUEST_INTERVAL = 1.0  # 秒
# ===== HMDB51类别名称 =====
# CLASS_NAMES = ["brush hair", "cartwheel", "catch", "chew", "clap", "climb", "climb stairs", "dive", "draw sword", "dribble", "drink", "eat", "fall floor", "fencing", "flic flac", "golf", "handstand", "hit", "hug", "jump", "kick", "kick ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride bike", "ride horse", "run", "shake hands", "shoot ball", "shoot bow", "shoot gun", "sit", "situp", "smile", "smoke", "somersault", "stand", "swing baseball", "sword", "sword exercise", "talk", "throw", "turn", "walk", "wave"]


# ===== UCF101类别名称 =====
CLASS_NAMES = [
    "Apply Eye Makeup", "Apply Lipstick", "Archery", "Baby Crawling", "Balance Beam", 
    "Band Marching", "Baseball Pitch", "Basketball", "Basketball Dunk", "Bench Press", 
    "Biking", "Billiards", "Blow Dry Hair", "Blowing Candles", "Body Weight Squats", 
    "Bowling", "Boxing Punching Bag", "Boxing Speed Bag", "Breast Stroke", "Brushing Teeth", 
    "Clean And Jerk", "Cliff Diving", "Cricket Bowling", "Cricket Shot", "Cutting In Kitchen", 
    "Diving", "Drumming", "Fencing", "Field Hockey Penalty", "Floor Gymnastics", 
    "Frisbee Catch", "Front Crawl", "Golf Swing", "Haircut", "Hammer Throw", 
    "Hammering", "Handstand Pushups", "Handstand Walking", "Head Massage", "High Jump", 
    "Horse Race", "Horse Riding", "Hula Hoop", "Ice Dancing", "Javelin Throw", 
    "Juggling Balls", "Jump Rope", "Jumping Jack", "Kayaking", "Knitting", 
    "Long Jump", "Lunges", "Military Parade", "Mixing", "Mopping Floor", 
    "Nunchucks", "Parallel Bars", "Pizza Tossing", "Playing Cello", "Playing Daf", 
    "Playing Dhol", "Playing Flute", "Playing Guitar", "Playing Piano", "Playing Sitar", 
    "Playing Tabla", "Playing Violin", "Pole Vault", "Pommel Horse", "Pull Ups", 
    "Punch", "Push Ups", "Rafting", "Rock Climbing Indoor", "Rope Climbing", 
    "Rowing", "Salsa Spin", "Shaving Beard", "Shotput", "Skate Boarding", 
    "Skiing", "Skijet", "Sky Diving", "Soccer Juggling", "Soccer Penalty", 
    "Still Rings", "Sumo Wrestling", "Surfing", "Swing", "Table Tennis Shot", 
    "Tai Chi", "Tennis Swing", "Throw Discus", "Trampoline Jumping", "Typing", 
    "Uneven Bars", "Volleyball Spiking", "Walking With Dog", "Wall Pushups", 
    "Writing On Board", "Yo Yo"
]

# =====Kinetics400类别名称 =====
# CLASS_NAMES = ["abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel", "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)", "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing cards", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ", "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting", "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving", "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "strumming guitar", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi", "taking a shower", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", "tossing coin", "tossing salad", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck", "using computer", "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"]

# ===== Something-Something V2 类别名称 =====
# CLASS_NAMES = ["Approaching something with your camera", "Attaching something to something", "Bending something so that it deforms", "Bending something until it breaks", "Burying something in something", "Closing something", "Covering something with something", "Digging something out of something", "Dropping something behind something", "Dropping something in front of something", "Dropping something into something", "Dropping something next to something", "Dropping something onto something", "Failing to put something into something because something does not fit", "Folding something", "Hitting something with something", "Holding something", "Holding something behind something", "Holding something in front of something", "Holding something next to something", "Holding something over something", "Laying something on the table on its side, not upright", "Letting something roll along a flat surface", "Letting something roll down a slanted surface", "Letting something roll up a slanted surface, so it rolls back down", "Lifting a surface with something on it but not enough for it to slide down", "Lifting a surface with something on it until it starts sliding down", "Lifting something up completely without letting it drop down", "Lifting something up completely, then letting it drop down", "Lifting something with something on it", "Lifting up one end of something without letting it drop down", "Lifting up one end of something, then letting it drop down", "Moving away from something with your camera", "Moving part of something", "Moving something across a surface until it falls down", "Moving something across a surface without it falling down", "Moving something and something away from each other", "Moving something and something closer to each other", "Moving something and something so they collide with each other", "Moving something and something so they pass each other", "Moving something away from something", "Moving something away from the camera", "Moving something closer to something", "Moving something down", "Moving something towards the camera", "Moving something up", "Opening something", "Picking something up", "Piling something up", "Plugging something into something", "Plugging something into something but pulling it right out as you remove your hand", "Poking a hole into some substance", "Poking a hole into something soft", "Poking a stack of something so the stack collapses", "Poking a stack of something without the stack collapsing", "Poking something so it slightly moves", "Poking something so lightly that it doesn't or almost doesn't move", "Poking something so that it falls over", "Poking something so that it spins around", "Pouring something into something", "Pouring something into something until it overflows", "Pouring something onto something", "Pouring something out of something", "Pretending or failing to wipe something off of something", "Pretending or trying and failing to twist something", "Pretending to be tearing something that is not tearable", "Pretending to close something without actually closing it", "Pretending to open something without actually opening it", "Pretending to pick something up", "Pretending to poke something", "Pretending to pour something out of something, but something is empty", "Pretending to put something behind something", "Pretending to put something into something", "Pretending to put something next to something", "Pretending to put something on a surface", "Pretending to put something onto something", "Pretending to put something underneath something", "Pretending to scoop something up with something", "Pretending to spread air onto something", "Pretending to sprinkle air onto something", "Pretending to squeeze something", "Pretending to take something from somewhere", "Pretending to take something out of something", "Pretending to throw something", "Pretending to turn something upside down", "Pulling something from behind of something", "Pulling something from left to right", "Pulling something from right to left", "Pulling something onto something", "Pulling something out of something", "Pulling two ends of something but nothing happens", "Pulling two ends of something so that it gets stretched", "Pulling two ends of something so that it separates into two pieces", "Pushing something from left to right", "Pushing something from right to left", "Pushing something off of something", "Pushing something onto something", "Pushing something so it spins", "Pushing something so that it almost falls off but doesn't", "Pushing something so that it falls off the table", "Pushing something so that it slightly moves", "Pushing something with something", "Putting number of something onto something", "Putting something and something on the table", "Putting something behind something", "Putting something in front of something", "Putting something into something", "Putting something next to something", "Putting something on a flat surface without letting it roll", "Putting something on a surface", "Putting something on the edge of something so it is not supported and falls down", "Putting something onto a slanted surface but it doesn't glide down", "Putting something onto something", "Putting something onto something else that cannot support it so it falls down", "Putting something similar to other things that are already on the table", "Putting something that can't roll onto a slanted surface, so it slides down", "Putting something that can't roll onto a slanted surface, so it stays where it is", "Putting something that cannot actually stand upright upright on the table, so it falls on its side", "Putting something underneath something", "Putting something upright on the table", "Putting something, something and something on the table", "Removing something, revealing something behind", "Rolling something on a flat surface", "Scooping something up with something", "Showing a photo of something to the camera", "Showing something behind something", "Showing something next to something", "Showing something on top of something", "Showing something to the camera", "Showing that something is empty", "Showing that something is inside something", "Something being deflected from something", "Something colliding with something and both are being deflected", "Something colliding with something and both come to a halt", "Something falling like a feather or paper", "Something falling like a rock", "Spilling something behind something", "Spilling something next to something", "Spilling something onto something", "Spinning something so it continues spinning", "Spinning something that quickly stops spinning", "Spreading something onto something", "Sprinkling something onto something", "Squeezing something", "Stacking number of something", "Stuffing something into something", "Taking one of many similar things on the table", "Taking something from somewhere", "Taking something out of something", "Tearing something into two pieces", "Tearing something just a little bit", "Throwing something", "Throwing something against something", "Throwing something in the air and catching it", "Throwing something in the air and letting it fall", "Throwing something onto a surface", "Tilting something with something on it slightly so it doesn't fall down", "Tilting something with something on it until it falls off", "Tipping something over", "Tipping something with something in it over, so something in it falls out", "Touching (without moving) part of something", "Trying but failing to attach something to something because it doesn't stick", "Trying to bend something unbendable so nothing happens", "Trying to pour something into something, but missing so it spills next to it", "Turning something upside down", "Turning the camera downwards while filming something", "Turning the camera left while filming something", "Turning the camera right while filming something", "Turning the camera upwards while filming something", "Twisting (wringing) something wet until water comes out", "Twisting something", "Uncovering something", "Unfolding something", "Wiping something off of something"]

# ===== Prompt 模板 =====
prompt_template = ChatPromptTemplate.from_template(
    """Describe the visual characteristics of the action "{class_name}" from the {dataset_name} video action recognition dataset.
Provide {n_words} single descriptive words (NOT phrases) about the motion, posture, and movement patterns.

Examples of good words: running, jumping, bending, upward, smooth, fast
Examples of bad responses: "hand moves up", "person is running"

Format your response as follow:\nword\nword\nword..."""
)

# prompt_template = ChatPromptTemplate.from_template(
#     """Please provide only {n_words} descriptive English words for the {dataset_name} {class_name} dataset. Format your response as follow:\nword\nword\nword..."""
# )
parser = StrOutputParser()

# ===== 清洗与解析工具 =====
LEADING_SYMBOLS = re.compile(r"^[\s\d\.\-\)\(]+")


def clean_and_validate_word(raw_word: str) -> str:
    word = raw_word.strip()
    word = LEADING_SYMBOLS.sub("", word)
    word = word.strip("\"' ")
    if not word:
        return ""
    normalized = word.replace("_", "-")
    test_token = normalized.replace("-", "")
    if not test_token.isalpha():
        return ""
    return normalized


def parse_llm_output(raw_text: str, target_count: int) -> List[str]:
    lines = raw_text.strip().splitlines()
    cleaned: List[str] = []
    seen = set()

    for line in lines:
        word = clean_and_validate_word(line)
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        cleaned.append(word)
        seen.add(key)

    if len(cleaned) > target_count:
        cleaned = cleaned[:target_count]
    elif len(cleaned) < target_count:
        print(f"⚠️ 仅获取 {len(cleaned)} 个词，将按实际数量保存。")

    return cleaned


def generate_attributes_for_class(class_name: str) -> List[str]:
    prompt_chain = prompt_template | model | parser
    response = prompt_chain.invoke(
        {
            "n_words": WORDS_PER_CLASS,
            "dataset_name": DATASET_NAME,
            "class_name": class_name,
        }
    )
    return parse_llm_output(response, WORDS_PER_CLASS)


def save_attributes(attributes: Dict[str, List[str]]) -> None:
    output_dir = Path("attributes")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{DATASET_NAME.lower()}_attributes.json"

    payload = {
        "dataset": DATASET_NAME,
        "num_classes": len(attributes),
        "words_per_class": WORDS_PER_CLASS,
        "attributes": attributes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n💾 属性词表已保存到: {output_path}")


def main() -> None:
    print("开始生成类别属性词表（单次预处理）...")
    print("=" * 60)
    print(f"数据集: {DATASET_NAME}")
    print(f"类别数: {len(CLASS_NAMES)}")
    print(f"每类目标词数: {WORDS_PER_CLASS}\n")

    attributes: Dict[str, List[str]] = {}

    for idx, cls in enumerate(CLASS_NAMES, 1):
        print(f"[{idx:03d}/{len(CLASS_NAMES):03d}] 处理类别: {cls}")
        try:
            words = generate_attributes_for_class(cls)
            attributes[cls] = words
            print(f"    ✓ 获得 {len(words)} 个词")
        except Exception as exc:
            print(f"    ✗ 生成失败: {exc}")
            attributes[cls] = []
        time.sleep(REQUEST_INTERVAL)

    save_attributes(attributes)


if __name__ == "__main__":
    main()
