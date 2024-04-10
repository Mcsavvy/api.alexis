"""Generate a random thread title for a chat thread."""
from random import choice

Adjectives = [
    'adorable', 'adventurous', 'agreeable', 'alert', 'alive', 'amused',
    'attractive', 'beautiful', 'better', 'bewildered', 'black', 'blue',
    'blue-eyed', 'blushing', 'brainy', 'brave', 'breakable', 'bright',
    'busy', 'calm', 'careful', 'cautious', 'charming', 'cheerful', 'clean',
    'clear', 'clever', 'cloudy', 'colorful', 'combative', 'comfortable',
    'cooperative', 'courageous', 'crazy', 'creepy', 'curious', 'cute',
    'defiant', 'delightful', 'determined', 'different', 'difficult',
    'distinct', 'dizzy', 'doubtful', 'eager', 'easy', 'elated', 'elegant',
    'enchanting', 'encouraging', 'energetic', 'enthusiastic', 'excited',
    'expensive', 'exuberant', 'fair', 'faithful', 'famous', 'fancy',
    'fantastic', 'fierce', 'fine', 'fragile', 'frail', 'frantic',
    'friendly', 'funny', 'gentle', 'gifted', 'glamorous', 'gleaming',
    'glorious', 'good', 'gorgeous', 'graceful', 'handsome', 'happy',
    'healthy', 'helpful', 'helpless', 'hilarious', 'homely', 'horrible',
    'important', 'impossible', 'inexpensive', 'innocent', 'inquisitive',
    'itchy', 'jealous', 'jittery', 'jolly', 'joyous', 'kind', 'lazy', 'light',
    'lively', 'lonely', 'long', 'lovely', 'lucky', 'magnificent', 'misty',
    'modern', 'motionless', 'muddy', 'mushy', 'mysterious', 'nasty',
    'naughty', 'nervous', 'nice', 'nutty', 'obedient', 'open', 'outrageous',
    'outstanding', 'perfect', 'plain', 'pleasant', 'poised', 'powerful',
    'precious', 'prickly', 'proud', 'puzzled', 'quaint', 'real', 'relieved',
    'repulsive', 'rich', 'scary', 'selfish', 'shiny', 'shy', 'silly', 'sleepy',
    'smiling', 'smoggy', 'sore', 'sparkling', 'splendid', 'spotless', 'stormy',
    'strange', 'successful', 'super', 'talented', 'tame', 'tasty', 'tender',
    'tense', 'terrible', 'thankful', 'thoughtful', 'thoughtless', 'tired',
    'tough', 'ugly', 'unsightly', 'unusual', 'uptight', 'vast', 'victorious',
    'vivacious', 'wandering', 'wide-eyed', 'wild', 'witty', 'worried',
    'worrisome', 'zany', 'zealous'
]

Nouns = [
    "algorithm", "array", "backup", "bandwidth",
    "binary", "blockchain", "blog", "browser", "buffer",
    "byte", "cache", "cloud", "compiler", "constant", "container",
    "database", "deployment", "device", "disk", "domain", "driver",
    "email", "engine", "firewall", "firmware", "framework",
    "gateway", "git", "graphics", "grid", "hardware", "host",
    "ide", "index", "infrastructure", "input", "instance", "internet",
    "intranet", "ip", "kernel", "key", "library", "link", "load",
    "loop", "machine", "malware", "memory", "middleware", "module",
    "monitor", "motherboard", "mouse", "network", "node", "object",
    "os", "output", "packet", "page", "path", "platform", "plugin",
    "port", "processor", "protocol", "query", "queue", "router",
    "runtime", "script", "sdk", "server", "service", "session",
    "shell", "software", "stack"
]

def gen_name() -> str:
    """Generate a random thread title."""
    return f"{choice(Adjectives)}-{choice(Nouns)}"
