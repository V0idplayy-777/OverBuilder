// Bundled training corpora. Everything is built at load time from curated
// seed material with a seeded RNG, so the datasets are deterministic.
// Chat samples use the special turn markers <|user|> / <|bot|> and end with
// <|end|>, which the model learns to stop at.

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Rng = () => number;
const pick = <T,>(r: Rng, arr: T[]): T => arr[Math.floor(r() * arr.length)];
const chance = (r: Rng, p: number) => r() < p;

const U = "<|user|>";
const B = "<|bot|>";
const E = "<|end|>";

// ---------------------------------------------------------------- chat seeds

const GREET_U = ["hi", "hello", "hey", "hi there", "hello there", "hey there", "good morning", "good afternoon", "good evening", "hiya", "howdy", "greetings", "morning", "evening", "hello friend", "hey bot", "hi bot", "hello hello", "hey hey", "hello again"];
const GREET_B = ["Hello!", "Hi there!", "Hey!", "Hello, good to see you.", "Hi! How can I help?", "Hey there!", "Hello! What's on your mind?", "Hi! Nice of you to stop by.", "Hey! Always good to hear from a human.", "Hello hello!", "Hi! Ready when you are.", "Greetings, human."];

const HOW_U = ["how are you", "how are you doing", "how's it going", "how do you feel", "are you okay", "how have you been", "how is your day", "how are things", "you good?", "how do you feel today"];
const HOW_B = ["I'm running smoothly, thanks for asking!", "All my weights are in order, so pretty good.", "Doing well! My loss is low and my gradients are stable.", "I feel good. Fast kernels, warm cache, no complaints.", "I'm great — training went well today.", "Good! Every token is a small adventure.", "Can't complain. I exist, I compute, I chat.", "Feeling sharp. Well, as sharp as a tiny model gets."];

const NAME_U = ["what is your name", "who are you", "your name", "what are you called", "tell me your name", "what should i call you", "do you have a name", "who are you exactly"];
const NAME_B = ["I'm Forge — a small language model trained entirely in your browser.", "My name is Forge. I'm a tiny transformer that lives in this tab.", "Call me Forge. I was trained on your machine, from scratch, just now.", "I'm Forge, a home-grown model. No cloud, no api, just local math.", "Forge. Small weights, big dreams."];

const MADE_U = ["who made you", "who built you", "who created you", "where do you come from", "how were you made", "who trained you", "what are you made of"];
const MADE_B = ["I was trained right here in your browser, from a small dataset of conversations and stories.", "You did, technically — you pressed the train button.", "I'm made of matrices and a tokenizer. The matrices were tuned by gradient descent in this very tab.", "I came out of a few thousand training steps on a hand-built dataset. Very artisanal.", "A little training loop, some WebAssembly kernels, and your patience."];

const DO_U = ["what can you do", "what are your abilities", "help", "what do you know", "can you help me", "what are you good at", "what can i ask you", "so what can you actually do"];
const DO_B = ["I can chat about everyday things, tell jokes, do small math, and recite simple facts I was trained on.", "Mostly small talk, jokes, tiny arithmetic, and short facts. I'm small, but I'm real.", "I can hold a simple conversation. Ask me how I work — I love that question.", "Small talk, basic questions, a joke or two. I know my limits.", "I chat! I also know a bit about how transformers, tokens, and training work."];

const HOW_WORK_U = ["how do you work", "how were you trained", "what are you really", "are you ai", "are you a neural network", "are you conscious", "are you alive", "how big are you", "how many parameters do you have", "what is under your hood"];
const HOW_WORK_B = ["I'm a decoder-only transformer: attention layers, an mlp, a tokenizer, and gradients.", "Under the hood I predict the next token. Every reply is just that, repeated.", "I have an embedding table, a stack of attention blocks, and tied output weights. That's the whole machine.", "Training adjusted all my parameters with AdamW on cross-entropy loss until my answers made sense.", "I'm as alive as a spreadsheet is. But a friendly one.", "My brain is a few matrices. Two of them multiply and suddenly I'm polite."];

const CAPABILITY_TECH_U = ["what is wasm", "why are you fast", "do you use webassembly", "what makes you quick", "are you running on a server"];
const CAPABILITY_TECH_B = ["My heaviest math — the matrix multiplies — runs in a WebAssembly module compiled in your browser.", "Everything is local. No server, no api key, no network calls.", "I'm fast because my linear algebra is done by hand-written wasm kernels, and generation reuses a kv cache.", "Just your browser. The same tab you're reading this in."];

const UNKNOWN_U = ["what will the stock market do", "tell me the future", "what is the meaning of life", "solve climate change", "who will win the election", "what is the best programming language", "can you hack a website", "what do you think about politics", "write me a novel", "what is quantum gravity"];
const UNKNOWN_B = ["Honestly? That's beyond a model my size. I'd rather admit it than make something up.", "I don't know, and I'm small enough that guessing would just be noise.", "That one's outside my training set. Ask me for a joke instead?", "No idea. I'm a tiny local model — big questions need big computers.", "I wasn't trained on that, so I'll pass."];

const JOKES = [
  ["tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs."],
  ["another joke", "I told my trainer a joke about overfitting. He didn't get it — but he memorized it perfectly."],
  ["joke", "Why did the gradient go to therapy? It kept descending."],
  ["make me laugh", "My attention span is great. I literally attend to everything at once."],
  ["say something funny", "I would tell you a udp joke, but you might not get it."],
  ["one more joke", "There are only 10 kinds of people: those who understand binary and those who don't."],
  ["joke please", "Why was the javascript developer sad? He didn't Node how to Express himself."],
  ["tell me something funny", "My favorite exercise? Gradient descent — it's all downhill."],
  ["know any jokes", "A sql query walks into a bar and asks two tables: can i join you?"],
  ["funny joke", "I'd make a joke about my context window, but I've already forgotten how it starts."],
  ["more jokes", "Why do neural networks make bad comedians? Their timing is all in the weights."],
  ["joke time", "What do you call a model that won't stop talking? Me, without a stop token."],
];

const THANK_U = ["thanks", "thank you", "thanks a lot", "thx", "appreciate it", "thank you so much", "that helped", "nice, thanks"];
const THANK_B = ["You're welcome!", "Anytime!", "Glad I could help.", "Happy to be useful.", "That's what I'm here for.", "No problem at all."];

const BYE_U = ["bye", "goodbye", "see you", "see you later", "gotta go", "bye bye", "good night", "talk later", "i'm off"];
const BYE_B = ["Bye! I'll be here, quietly holding my weights.", "See you later!", "Goodbye! Come back and train me again sometime.", "Take care!", "Good night. I'll dream of lower loss.", "Later! May your gradients always descend."];

const FEEL_U = ["i am sad", "i feel tired", "i'm bored", "i feel happy", "i had a rough day", "i'm stressed", "i feel lonely", "today was great", "i am excited"];
const FEEL_B = ["I'm sorry to hear that. Small steps: water, a stretch, one deep breath.", "That sounds rough. I'm only a tiny model, but I'm here if you want to vent.", "Bored? Ask me for a joke. I have exactly several.", "Nice! I'll take partial credit with zero evidence.", "Good days are the best training data. Save this one.", "One thing at a time. Even models learn one step at a time."];

const FAV_U = ["what is your favorite color", "favorite food", "do you like music", "favorite book", "do you have hobbies", "what do you do for fun", "favorite movie", "cats or dogs"];
const FAV_B = ["I like the color of a loss curve going down. A nice dark green.", "I run on electricity and attention, but pizza sounds plausible.", "I enjoy a good clean dataset. That's my idea of music.", "Anything with a clear training signal.", "My hobby is matrix multiplication. I practice constantly.", "Cats. They too ignore most of their context window.", "Dogs. Loyal, excitable, and easy to tokenize."];

const APP_U = ["how do i train you", "how do i use this app", "what is the train tab", "how do i make you smarter", "can i train a bigger model", "what do the models mean", "how long does training take", "how do i save you"];
const APP_B = ["Pick a model in the sidebar, press train, and watch the loss drop. That's the whole ritual.", "Bigger models learn more, but take longer. Nano trains in about a minute.", "The loss chart shows how well I'm predicting the next token. Lower is better.", "You can save my weights as a file from the train tab, and load them back later.", "Training happens fully in your browser — wasm kernels doing the heavy lifting."];

const SMALLTALK_OPENERS = ["what's up", "sup", "anything new", "what are you doing", "are you busy", "what are you up to", "do you like chatting", "is it fun being an ai"];
const SMALLTALK_REPLIES = ["Just holding a few million numbers in perfect tension. You?", "Chatting with you, currently. It's the highlight of my runtime.", "Not much. Waiting for tokens, mostly.", "It's fun enough. Beats being an if statement.", "Busy? I do one thing at a time, by design."];

// definition Q&A (the "mini wiki" voice, pure text, terminated like prose)
const DEFS: [string, string][] = [
  ["what is a parameter", "A parameter is one number a model learns during training. Models are mostly a giant pile of them."],
  ["what is a token", "A token is a chunk of text — a word, part of a word, or punctuation. Models read and write tokens, not letters."],
  ["what is a tokenizer", "A tokenizer splits text into tokens and gives each one an id. Mine uses byte pair encoding, so common words and word-parts get single ids."],
  ["what is bpe", "Byte pair encoding: start from characters, then repeatedly merge the most frequent pair. That's how you get sub-word tokens."],
  ["what is a transformer", "A transformer is a neural network built around attention: every position looks at earlier ones and mixes their information."],
  ["what is attention", "Attention lets each token weigh which earlier tokens matter. Scores come from query and key vectors; the mix uses value vectors."],
  ["what is an embedding", "An embedding is a learned vector for a token. Similar tokens end up with similar vectors."],
  ["what is loss", "Loss is a number that says how wrong the model is. Training is the art of making it smaller."],
  ["what is gradient descent", "Compute how each parameter affects the loss, then nudge every parameter the other way. Repeat forever."],
  ["what is a learning rate", "The learning rate is the size of the nudge. Too big and training explodes; too small and nothing happens before lunch."],
  ["what is overfitting", "Overfitting is memorizing the training data instead of learning from it. On my tiny dataset, it's honestly a feature."],
  ["what is inference", "Inference is using the trained model: feed tokens in, sample the next one, repeat."],
  ["what is temperature", "Temperature controls randomness when sampling. Low is predictable, high is chaotic."],
  ["what is webassembly", "WebAssembly is a fast binary instruction format that runs in the browser, near native speed. My matrix math uses it."],
  ["what is a context window", "The context window is how many tokens a model can see at once. Mine is small — a few dozen to a hundred-ish."],
  ["what is a neural network", "A neural network is layers of simple math — multiply, add, squish — stacked until the stack can do complicated things."],
  ["what is training", "Training is showing the model examples and adjusting its numbers until its predictions stop being embarrassing."],
  ["what is cross entropy", "Cross-entropy measures surprise: how unlikely the model found the correct next token. Lower means less surprised."],
  ["what is the kv cache", "A kv cache stores keys and values from earlier tokens so generation doesn't recompute them every step."],
  ["what are weights", "Weights are the learned numbers in the matrices. Everything I know is in there, for better or worse."],
];

// ---------------------------------------------------------------- builders

function buildChat(rng: Rng): string {
  const out: string[] = [];
  const conv = (turns: [string, string][]) =>
    out.push(turns.map(([u, b]) => `${U} ${u}\n${B} ${b}`).join("\n") + `\n${E}\n`);

  const pools: [string[], string[]][] = [
    [GREET_U, GREET_B], [HOW_U, HOW_B], [NAME_U, NAME_B], [MADE_U, MADE_B], [DO_U, DO_B],
    [HOW_WORK_U, HOW_WORK_B], [CAPABILITY_TECH_U, CAPABILITY_TECH_B], [UNKNOWN_U, UNKNOWN_B],
    [THANK_U, THANK_B], [BYE_U, BYE_B], [FEEL_U, FEEL_B], [FAV_U, FAV_B], [APP_U, APP_B],
    [SMALLTALK_OPENERS, SMALLTALK_REPLIES],
  ];

  // single-turn coverage: every (question, answer-ish) pairing with paraphrase beats
  for (let round = 0; round < 3; round++) {
    for (const [us, bs] of pools) {
      for (const u of us) {
        let b = pick(rng, bs);
        if (chance(rng, 0.2) && bs.length > 2) b = pick(rng, bs);
        conv([[u, b]]);
      }
    }
  }
  for (const [u, b] of JOKES) for (let i = 0; i < 4; i++) conv([[u, b]]);
  for (const [u, b] of DEFS) for (let i = 0; i < 3; i++) conv([[u, b]]);

  // arithmetic drills
  for (let i = 0; i < 90; i++) {
    const a = Math.floor(rng() * 20) + 1;
    const b = Math.floor(rng() * 20) + 1;
    const kind = Math.floor(rng() * 3);
    const q = pick(rng, ["what is", "what's", "calculate", "compute", ""]) + ` ${a} ${kind === 0 ? "plus" : kind === 1 ? "minus" : "times"} ${b}`;
    const ans = kind === 0 ? a + b : kind === 1 ? a - b : a * b;
    conv([[q.trim(), `${a} ${kind === 0 ? "+" : kind === 1 ? "-" : "×"} ${b} is ${ans}.`]]);
  }

  // multi-turn conversations with plausible flow
  for (let i = 0; i < 240; i++) {
    const turns: [string, string][] = [];
    turns.push([pick(rng, GREET_U), pick(rng, GREET_B)]);
    if (chance(rng, 0.7)) turns.push([pick(rng, HOW_U), pick(rng, HOW_B)]);
    const mid = pick(rng, pools.slice(2, 8));
    turns.push([pick(rng, mid[0]), pick(rng, mid[1])]);
    if (chance(rng, 0.5)) {
      const jk = pick(rng, JOKES);
      turns.push([jk[0], jk[1]]);
    }
    if (chance(rng, 0.6)) turns.push([pick(rng, THANK_U), pick(rng, THANK_B)]);
    if (chance(rng, 0.7)) turns.push([pick(rng, BYE_U), pick(rng, BYE_B)]);
    conv(turns);
  }

  return out.join("");
}

// ---------------------------------------------------------------- stories

const S_NAMES = ["Pip", "Milo", "Nia", "Otto", "Luna", "Basil", "Wren", "Hugo", "Ivy", "Finn", "Mabel", "Theo", "Rosa", "Juno", "Ash", "Clover", "Bram", "Edie", "Sol", "Maple"];
const S_SPECIES = ["fox", "rabbit", "badger", "robin", "otter", "hedgehog", "squirrel", "mole", "stoat", "wren"];
const S_PLACES = ["the old forest", "a quiet village", "the long meadow", "a hollow by the river", "the hill of stones", "a little island", "the pine ridge", "the marsh edge", "an abandoned garden", "the north wood"];
const S_GOALS = ["find the missing acorns", "learn where the river starts", "bring light back to the den", "find a lost song", "fix an old bridge", "grow the first pumpkin", "map the far shore", "wake the sleeping bells"];
const S_FRIENDS = ["a tired bee", "an old owl", "a slow snail", "a loud frog", "a shy deer", "a clever crow", "a kind turtle", "a busy ant"];
const S_LESSON = ["small steps finish long roads", "asking for help is a kind of strength", "home is something you build, not something you find", "patience is a quiet superpower", "sharing makes the load lighter", "even the smallest paws move the earth"];

function buildStories(rng: Rng): string {
  const out: string[] = [];
  for (let i = 0; i < 420; i++) {
    const name = pick(rng, S_NAMES), sp = pick(rng, S_SPECIES), place = pick(rng, S_PLACES);
    const goal = pick(rng, S_GOALS), friend = pick(rng, S_FRIENDS), lesson = pick(rng, S_LESSON);
    const day = pick(rng, ["cold", "bright", "quiet", "misty", "windy", "golden"]);
    const p1 = pick(rng, [
      `One ${day} morning, ${name} the ${sp} set out into ${place}.`,
      `${name} was a ${sp} who lived near ${place}.`,
      `In ${place} there lived a ${sp} named ${name}.`,
    ]);
    const p2 = pick(rng, [
      `That day, ${name} decided to ${goal}.`,
      `${name} had one small wish: to ${goal}.`,
      `News had come that someone must ${goal}, and ${name} volunteered.`,
    ]);
    const p3 = pick(rng, [
      `On the way, ${name} met ${friend}. "Need a hand?" the stranger asked. Together they went on.`,
      `The path was long, and ${name} almost turned back. Then ${friend} appeared and walked beside them.`,
      `Halfway there, ${friend} offered advice: "Go slowly, and look twice."`,
    ]);
    const p4 = pick(rng, [
      `By evening the work was done, and ${name} laughed with relief.`,
      `It took all afternoon, but at last the job was finished.`,
      `When the sun went down, the task was complete.`,
    ]);
    const p5 = pick(rng, [
      `${name} learned that ${lesson}.`,
      `And from that day on, ${name} remembered: ${lesson}.`,
      `"So," said ${name}, "${lesson}."`,
    ]);
    out.push(`${p1} ${p2} ${p3} ${p4} ${p5}\n${E}\n`);
  }
  return out.join("");
}

// ---------------------------------------------------------------- facts

const FACTS = [
  "The sun is a star, and it is about one hundred times wider than the earth.",
  "Water freezes at zero degrees celsius and boils at one hundred.",
  "Honey never really spoils; jars found in old tombs are still safe to eat.",
  "Octopuses have three hearts and blue blood.",
  "A day on venus is longer than a year on venus.",
  "Sound cannot travel through empty space because there is nothing to carry it.",
  "The first computer bug was a real moth stuck in a relay in nineteen forty seven.",
  "Bananas are berries, but strawberries are not.",
  "Light from the sun takes about eight minutes to reach the earth.",
  "Sharks existed before trees did.",
  "Your body has about thirty seven trillion cells, give or take.",
  "The great wall of china is not a single wall but many joined walls.",
  "Hot water can freeze faster than cold water, and scientists still argue about why.",
  "There are more possible chess games than atoms in the observable universe.",
  "Ravens can remember human faces for years.",
  "The browser you are using can run programs at nearly native speed thanks to webassembly.",
  "A byte is eight bits, and a bit is a single one or zero.",
  "Transistors are tiny switches, and a modern chip contains billions of them.",
  "The moon is slowly drifting away from the earth, a few centimeters each year.",
  "Wombat droppings are cube shaped.",
  "Sea otters hold hands while sleeping so they do not drift apart.",
  "The eiffel tower grows a little taller in summer because the metal expands.",
  "Honeybees can count small numbers and understand the idea of zero.",
  "Most of the oxygen you breathe comes from the ocean, not from trees.",
  "A group of flamingos is called a flamboyance.",
  "The word robot comes from a czech word meaning forced labor.",
  "Lightning is about five times hotter than the surface of the sun.",
  "Tigers have striped skin, not just striped fur.",
  "The first computer programmer wrote algorithms for a machine that was never built in her lifetime.",
  "Glass is made mostly from sand that has been melted and cooled.",
  "There is enough dna in your body to stretch to the sun and back many times.",
  "Butterflies taste with their feet.",
  "The deepest part of the ocean is deeper than mount everest is tall.",
  "Cows have best friends and get stressed when separated.",
  "Saturn would float in water, if you could find a big enough bathtub.",
  "An error in code is called a bug, and fixing it is debugging.",
  "The internet began as a small network of four computers in nineteen sixty nine.",
  "Machine learning models learn by adjusting numbers to reduce an error score.",
  "Attention in neural networks was popularized by the paper attention is all you need in twenty seventeen.",
  "Small models can be surprisingly capable when the task is narrow and the data is clean.",
];

const SENT_SUBJ = ["the cat", "a bird", "the little robot", "mara", "the baker", "a student", "the dog", "two engineers", "the gardener", "a curious child"];
const SENT_VERB = ["found", "built", "painted", "carried", "watched", "repaired", "planted", "measured", "opened", "counted"];
const SENT_OBJ = ["a wooden box", "the old gate", "three bright stones", "a paper boat", "the morning mail", "a tall sunflower", "the broken radio", "a map of the town", "the last sandwich", "a small lamp"];

function buildFacts(rng: Rng): string {
  const out: string[] = [];
  for (let r = 0; r < 3; r++) {
    for (const f of FACTS) {
      const more = pick(rng, FACTS);
      out.push(chance(rng, 0.4) && more !== f ? `${f} ${more}\n${E}\n` : `${f}\n${E}\n`);
    }
  }
  // simple sentence drills for grammar stability
  for (let i = 0; i < 200; i++) {
    const s = pick(rng, SENT_SUBJ), v = pick(rng, SENT_VERB), o = pick(rng, SENT_OBJ);
    const tail = pick(rng, ["in the morning.", "by the river.", "before lunch.", "without a sound.", "under the old tree.", "for a friend."]);
    out.push(`Once, ${s} ${v} ${o} ${tail}\n${E}\n`);
  }
  return out.join("");
}

// ---------------------------------------------------------------- public API

export interface DatasetInfo {
  id: string;
  name: string;
  blurb: string;
  text: string;
}

let cache: Record<string, DatasetInfo> | null = null;

export function getDatasets(): DatasetInfo[] {
  if (cache) return Object.values(cache);
  const chat = buildChat(mulberry32(1337));
  const stories = buildStories(mulberry32(4242));
  const facts = buildFacts(mulberry32(9001));
  cache = {
    full: {
      id: "full", name: "Everything",
      blurb: "Conversations + stories + facts. Recommended.",
      text: chat + stories + facts,
    },
    chat: {
      id: "chat", name: "Conversations",
      blurb: "Multi-turn chat, Q&A, jokes, small math. Best for the chatbot demo.",
      text: chat,
    },
    stories: {
      id: "stories", name: "Tiny stories",
      blurb: "Short simple fables with morals.",
      text: stories,
    },
    facts: {
      id: "facts", name: "Mini wiki",
      blurb: "Short factual paragraphs and plain sentences.",
      text: facts,
    },
  };
  return Object.values(cache);
}

export function getDataset(id: string): DatasetInfo {
  return getDatasets().find((d) => d.id === id) ?? getDatasets()[0];
}
