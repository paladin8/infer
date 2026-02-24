"""Serving benchmark: measure throughput, TTFT, and ITL against a running infer server.

This is a pure async HTTP client — no model/tokenizer/torch dependencies. It connects
to an already-running ``infer`` server, fires requests per workload arrival patterns,
parses SSE streams, collects per-token timestamps, and reports metrics.

Usage (single workload):
    uv run python benchmarks/bench_serving.py \
        --server http://localhost:8000 \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --workload baseline

Usage (all workloads):
    uv run python benchmarks/bench_serving.py \
        --server http://localhost:8000 \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --workload all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPORTS_DIR = Path(__file__).parent / "reports"

# Rough chars-per-token estimate for English text.  Actual token counts come
# from the server's ``usage`` field in the ``done`` SSE event.
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSpec:
    """What to send to the server."""

    prompt: str
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    send_delay_s: float = 0.0


@dataclass
class RequestResult:
    """Per-request timing and outcome."""

    request_idx: int
    prompt_tokens: int
    completion_tokens: int
    send_time_s: float
    first_token_time_s: float
    done_time_s: float
    token_timestamps_s: list[float] = field(default_factory=list)
    finish_reason: str = ""
    error: str | None = None

    @property
    def ttft_s(self) -> float:
        if self.first_token_time_s <= 0 or self.send_time_s <= 0:
            return 0.0
        return self.first_token_time_s - self.send_time_s

    @property
    def request_latency_s(self) -> float:
        if self.done_time_s <= 0 or self.send_time_s <= 0:
            return 0.0
        return self.done_time_s - self.send_time_s

    @property
    def itl_values_s(self) -> list[float]:
        if len(self.token_timestamps_s) < 2:
            return []
        return [
            self.token_timestamps_s[i] - self.token_timestamps_s[i - 1]
            for i in range(1, len(self.token_timestamps_s))
        ]


@dataclass
class WorkloadReport:
    """Aggregate metrics for a workload run."""

    workload_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    wall_clock_s: float
    throughput_output_tps: float
    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_mean_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    results: list[RequestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Arrival pattern helpers
# ---------------------------------------------------------------------------


def uniform_delays(n: int, rate_rps: float) -> list[float]:
    """Evenly spaced delays at the given request rate."""
    interval = 1.0 / rate_rps
    return [i * interval for i in range(n)]


def poisson_delays(n: int, rate_rps: float, rng: random.Random) -> list[float]:
    """Poisson process inter-arrival times (cumulative delays)."""
    delays: list[float] = []
    t = 0.0
    for _ in range(n):
        delays.append(t)
        t += rng.expovariate(rate_rps)
    return delays


def burst_delays(n: int, burst_size: int, burst_interval_s: float) -> list[float]:
    """Groups sent simultaneously, gap between groups."""
    delays: list[float] = []
    for i in range(n):
        group_idx = i // burst_size
        delays.append(group_idx * burst_interval_s)
    return delays


# ---------------------------------------------------------------------------
# Prompt corpus — fixed realistic text, no tokenizer dependency
# ---------------------------------------------------------------------------

_PASSAGES = [
    (
        "Photosynthesis is one of the most fundamental biological processes on Earth, "
        "serving as the primary mechanism by which solar energy is converted into chemical "
        "energy that sustains nearly all life forms. This process occurs predominantly in "
        "the chloroplasts of plant cells, where specialized pigments, most notably "
        "chlorophyll, capture photons from sunlight. The process unfolds in two major "
        "stages: the light-dependent reactions and the Calvin cycle. During the "
        "light-dependent reactions, which take place in the thylakoid membranes, water "
        "molecules are split through photolysis, releasing oxygen as a byproduct while "
        "generating ATP and NADPH. These energy carriers then power the Calvin cycle in "
        "the stroma, where carbon dioxide from the atmosphere is fixed into organic "
        "molecules through a series of enzyme-catalyzed reactions. The enzyme RuBisCO "
        "plays a central role in this carbon fixation, catalyzing the attachment of CO2 "
        "to ribulose bisphosphate."
    ),
    (
        "The history of computing stretches back millennia, from the abacus of ancient "
        "civilizations to the silicon chips that power modern devices. Charles Babbage "
        "conceived the Analytical Engine in the 1830s, a mechanical general-purpose "
        "computer that anticipated many features of modern machines. Ada Lovelace wrote "
        "what is considered the first algorithm intended for machine processing, making "
        "her the world's first computer programmer. The twentieth century brought "
        "electronic computing, starting with vacuum tube machines like ENIAC in 1945, "
        "which could perform thousands of calculations per second. The invention of the "
        "transistor at Bell Labs in 1947 revolutionized electronics, leading to smaller, "
        "faster, and more reliable computers. The integrated circuit, developed "
        "independently by Jack Kilby and Robert Noyce, placed multiple transistors on a "
        "single chip, paving the way for the microprocessor revolution of the 1970s."
    ),
    (
        "Ocean currents are continuous, directed movements of seawater driven by forces "
        "acting upon the water, including wind, the Coriolis effect, temperature "
        "differences, and salinity gradients. Surface currents, which make up about ten "
        "percent of all water in the ocean, are primarily driven by wind patterns. The "
        "Gulf Stream, one of the most well-known ocean currents, transports warm water "
        "from the Gulf of Mexico northeastward across the Atlantic, significantly "
        "moderating the climate of Western Europe. Deep ocean currents, also known as "
        "thermohaline circulation, are driven by differences in water density caused by "
        "variations in temperature and salinity. This global conveyor belt moves water "
        "slowly through the deep ocean basins, taking roughly a thousand years to "
        "complete one full circuit. Changes to these circulation patterns can have "
        "profound effects on global climate and weather systems."
    ),
    (
        "Ancient Roman engineering achievements continue to influence modern construction "
        "and infrastructure. The Romans perfected the use of concrete, developing a "
        "hydraulic cement that could set underwater, enabling the construction of ports, "
        "aqueducts, and structures that have endured for two millennia. The Pantheon in "
        "Rome, completed around 126 AD, features an unreinforced concrete dome that "
        "remains the largest of its kind in the world. Roman roads, built to connect the "
        "vast empire, employed sophisticated layered construction with drainage systems "
        "that many modern highways still emulate. The extensive aqueduct system "
        "transported fresh water over long distances using gravity alone, with some "
        "channels spanning hundreds of kilometers. Roman arches and vaults distributed "
        "weight efficiently, allowing the construction of massive structures like the "
        "Colosseum, which could seat fifty thousand spectators."
    ),
    (
        "Machine learning is a subset of artificial intelligence focused on building "
        "systems that learn from data rather than following explicitly programmed rules. "
        "Supervised learning, the most common paradigm, involves training models on "
        "labeled datasets where the correct output is known for each input. Neural "
        "networks, inspired by biological neurons, consist of layers of interconnected "
        "nodes that transform input data through weighted connections and nonlinear "
        "activation functions. Deep learning extends this concept with many layers, "
        "enabling the automatic extraction of hierarchical features from raw data. "
        "Gradient descent optimization adjusts model parameters iteratively to minimize "
        "a loss function that quantifies prediction errors. Regularization techniques "
        "such as dropout and weight decay help prevent overfitting, where a model "
        "memorizes training data rather than learning generalizable patterns. Transfer "
        "learning allows models pre-trained on large datasets to be fine-tuned for "
        "specific tasks with limited data."
    ),
    (
        "The water cycle describes the continuous movement of water within the Earth "
        "and atmosphere through processes including evaporation, condensation, "
        "precipitation, and runoff. Solar energy drives evaporation from oceans, lakes, "
        "and rivers, converting liquid water into water vapor that rises into the "
        "atmosphere. As this vapor ascends and cools, it condenses around tiny particles "
        "called condensation nuclei to form clouds. When cloud droplets combine and grow "
        "heavy enough, they fall as precipitation in the form of rain, snow, sleet, or "
        "hail. Some precipitation infiltrates the soil to recharge groundwater aquifers, "
        "while the remainder flows over the surface as runoff, eventually returning to "
        "rivers, lakes, and oceans. Transpiration from plants releases additional water "
        "vapor into the atmosphere, contributing significantly to local humidity and "
        "rainfall patterns. This cycle has operated continuously for billions of years."
    ),
    (
        "Musical harmony is the study of how individual notes combine simultaneously to "
        "produce chords and how chords progress in sequence to create musical phrases. "
        "Western tonal harmony is built on the foundation of the major and minor scales, "
        "each consisting of seven notes with specific interval patterns. Chords are "
        "typically constructed by stacking thirds, with triads consisting of a root, "
        "third, and fifth, and seventh chords adding an additional note. The relationship "
        "between the tonic chord and the dominant chord creates a sense of tension and "
        "resolution that drives harmonic motion. Cadences, which are standardized chord "
        "progressions that conclude musical phrases, provide structural punctuation "
        "similar to commas and periods in written language. Voice leading governs how "
        "individual notes within chords move to notes in subsequent chords, with smooth "
        "stepwise motion generally preferred over large leaps."
    ),
    (
        "Plate tectonics is the scientific theory explaining the large-scale motion of "
        "Earth's lithosphere, which is divided into several major and minor tectonic "
        "plates that float on the semi-fluid asthenosphere beneath. These plates move "
        "at rates of a few centimeters per year, driven by convection currents in the "
        "mantle and gravitational forces at subduction zones. At divergent boundaries, "
        "plates move apart and new crust is formed from rising magma, as seen at "
        "mid-ocean ridges like the Mid-Atlantic Ridge. At convergent boundaries, one "
        "plate slides beneath another in a process called subduction, creating deep "
        "ocean trenches and volcanic arcs. Transform boundaries occur where plates slide "
        "horizontally past each other, producing earthquakes along fault lines such as "
        "the San Andreas Fault in California. Continental drift, first proposed by "
        "Alfred Wegener in 1912, was confirmed by plate tectonic theory decades later."
    ),
    (
        "The principles of supply and demand form the foundation of microeconomic "
        "theory, describing how prices emerge from the interaction of buyers and sellers "
        "in competitive markets. The law of demand states that as the price of a good "
        "increases, the quantity demanded decreases, holding other factors constant. "
        "Conversely, the law of supply indicates that higher prices incentivize producers "
        "to offer more of a good. Market equilibrium occurs at the price where the "
        "quantity supplied equals the quantity demanded, with any deviation creating "
        "pressure for the price to return to equilibrium. Price elasticity measures how "
        "responsive quantity demanded or supplied is to price changes, with necessities "
        "tending to be inelastic and luxury goods more elastic. Government interventions "
        "such as price floors and ceilings can create surpluses or shortages by "
        "preventing the market from reaching its natural equilibrium."
    ),
    (
        "The human immune system is a complex network of cells, tissues, and organs that "
        "work together to defend the body against pathogens including bacteria, viruses, "
        "fungi, and parasites. The innate immune system provides immediate, nonspecific "
        "defense through physical barriers like skin and mucous membranes, as well as "
        "cellular components such as neutrophils and macrophages that engulf and destroy "
        "invaders. The adaptive immune system mounts targeted responses through "
        "lymphocytes: B cells produce antibodies that bind to specific antigens, while "
        "T cells directly attack infected cells or coordinate immune responses. Memory "
        "cells created during an initial infection provide lasting immunity, enabling a "
        "faster and stronger response upon subsequent exposure to the same pathogen. "
        "Vaccination exploits this memory mechanism by exposing the immune system to "
        "harmless forms of pathogens, priming it for future encounters."
    ),
    (
        "Stellar evolution describes the life cycle of stars from their formation in "
        "molecular clouds through their eventual death. Stars are born when regions of "
        "a nebula collapse under gravity, heating up until nuclear fusion ignites in the "
        "core, converting hydrogen into helium and releasing enormous energy. A star "
        "spends most of its life on the main sequence, where the outward pressure from "
        "fusion balances the inward pull of gravity. When hydrogen fuel in the core is "
        "exhausted, the star's fate depends on its mass: low-mass stars like our Sun "
        "expand into red giants before shedding their outer layers as planetary nebulae, "
        "leaving behind white dwarfs. Massive stars undergo more dramatic endings, "
        "fusing progressively heavier elements until iron accumulates in the core. The "
        "core then collapses catastrophically, producing a supernova explosion that can "
        "outshine an entire galaxy, leaving behind either a neutron star or black hole."
    ),
    (
        "Classical rhetoric, as codified by Aristotle, Cicero, and Quintilian, provides "
        "a systematic framework for persuasive communication that remains relevant in "
        "modern discourse. Aristotle identified three primary modes of persuasion: ethos "
        "appeals to the speaker's credibility and character, pathos engages the audience's "
        "emotions, and logos relies on logical reasoning and evidence. The five canons of "
        "rhetoric established by the Roman tradition encompass invention, arrangement, "
        "style, memory, and delivery, providing a comprehensive process for composing and "
        "presenting arguments. Logical fallacies, which are errors in reasoning that "
        "undermine the validity of an argument, include ad hominem attacks, straw man "
        "misrepresentations, false dilemmas, and appeals to authority. Understanding these "
        "principles helps both in constructing compelling arguments and in critically "
        "evaluating the arguments of others."
    ),
]

_INSTRUCTIONS = [
    "Based on the information above, provide a detailed explanation of the key concepts.",
    "Summarize the main points discussed in the preceding text.",
    "Analyze the relationships between the topics described above.",
    "Identify the most important principles mentioned and explain their significance.",
    "Compare and contrast the different processes or concepts described above.",
    "Explain how the concepts above apply to real-world scenarios.",
    "Describe the cause-and-effect relationships present in the text above.",
    "Evaluate the strengths and limitations of the approaches described.",
]

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant specialized in answering questions about "
    "science, history, technology, and the humanities. You provide detailed, accurate, "
    "and well-structured responses. When explaining complex topics, you break them down "
    "into clear logical steps. You cite relevant principles, theories, or historical "
    "context when applicable. Your tone is professional yet accessible, suitable for "
    "an educated general audience.\n\n"
    "When given a passage of text followed by a question or instruction, you should:\n"
    "1. Carefully read and understand the provided context.\n"
    "2. Identify the key themes, concepts, and relationships.\n"
    "3. Formulate a comprehensive response that directly addresses the question.\n"
    "4. Support your response with specific details from the provided text.\n"
    "5. Where appropriate, draw connections to broader concepts or related fields.\n\n"
    "Important guidelines:\n"
    "- Be thorough but concise. Avoid unnecessary repetition.\n"
    "- Use clear topic sentences and logical paragraph structure.\n"
    "- Define technical terms when they first appear.\n"
    "- Acknowledge uncertainty when the provided information is insufficient.\n"
    "- Maintain objectivity and present multiple perspectives when relevant.\n\n"
    "Remember that your primary goal is to help the user understand the material "
    "deeply, not merely to restate what was written. Add value through analysis, "
    "synthesis, and clear explanation."
)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def make_prompt(target_tokens: int, seed: int) -> str:
    """Build a realistic text prompt of approximately *target_tokens* length.

    Places the instruction first, then fills with context passages to reach the
    target.  This structure (task → context) is natural for instruction-tuned
    models and ensures the instruction is never truncated.

    Uses ``CHARS_PER_TOKEN`` as a rough sizing guide; the server returns
    authoritative token counts in its ``usage`` field.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    rng = random.Random(seed)

    instruction = rng.choice(_INSTRUCTIONS)

    shuffled = list(_PASSAGES)
    rng.shuffle(shuffled)

    # Build: instruction + context passages up to target length.
    context_parts: list[str] = []
    current_len = len(instruction) + 2  # +2 for "\n\n" after instruction
    idx = 0
    while current_len < target_chars:
        passage = shuffled[idx % len(shuffled)]
        context_parts.append(passage)
        current_len += len(passage) + 2  # +2 for "\n\n" separator
        idx += 1

    if context_parts:
        context = "\n\n".join(context_parts)
        text = instruction + "\n\n" + context
    else:
        text = instruction

    # Trim context (never the instruction) to approximate target.
    if len(text) > target_chars + 200:
        cut = text.rfind(" ", 0, target_chars + 100)
        if cut > len(instruction):
            text = text[:cut]

    return text


def make_shared_prefix_prompt(prefix_tokens: int, suffix_tokens: int, suffix_seed: int) -> str:
    """Shared system prompt prefix + unique suffix per request.

    The prefix is always identical across requests (deterministic, seed=0 filler).
    The suffix varies per request via *suffix_seed*.
    """
    prefix = _SYSTEM_PROMPT
    prefix_target_chars = prefix_tokens * CHARS_PER_TOKEN

    # Extend prefix with filler context if the system prompt is shorter than target.
    if len(prefix) < prefix_target_chars:
        filler = make_prompt(prefix_tokens, seed=0)
        prefix = prefix + "\n\n" + filler

    # Trim at word boundary.
    if len(prefix) > prefix_target_chars:
        cut = prefix.rfind(" ", 0, prefix_target_chars)
        if cut > 0:
            prefix = prefix[:cut]

    suffix = make_prompt(suffix_tokens, seed=suffix_seed)
    return prefix + "\n\n" + suffix


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------

WorkloadGenerator = Callable[[int, random.Random], list[RequestSpec]]


@dataclass(frozen=True)
class WorkloadDef:
    """A workload definition."""

    name: str
    description: str
    generator: WorkloadGenerator
    default_num_requests: int
    default_warmup_requests: int
    sequential: bool = False


def _gen_baseline(n: int, rng: random.Random) -> list[RequestSpec]:
    """Sequential single requests, fixed lengths."""
    specs: list[RequestSpec] = []
    for _ in range(n):
        prompt = make_prompt(256, seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=256, send_delay_s=0.0))
    return specs


def _gen_continuous_batching(n: int, rng: random.Random) -> list[RequestSpec]:
    """Staggered arrivals, varying lengths."""
    delays = uniform_delays(n, rate_rps=4.0)
    specs: list[RequestSpec] = []
    for i in range(n):
        target_tokens = rng.randint(64, 512)
        max_tokens = rng.randint(64, 256)
        prompt = make_prompt(target_tokens, seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=max_tokens, send_delay_s=delays[i]))
    return specs


def _gen_paged_attention(n: int, rng: random.Random) -> list[RequestSpec]:
    """Bursty arrivals, bimodal lengths."""
    delays = burst_delays(n, burst_size=8, burst_interval_s=0.5)
    specs: list[RequestSpec] = []
    for i in range(n):
        target_tokens = rng.randint(32, 128) if rng.random() < 0.5 else rng.randint(512, 1024)
        max_tokens = rng.randint(128, 512)
        prompt = make_prompt(target_tokens, seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=max_tokens, send_delay_s=delays[i]))
    return specs


def _gen_chunked_prefill(n: int, rng: random.Random) -> list[RequestSpec]:
    """Long prompts, Poisson arrivals."""
    delays = poisson_delays(n, rate_rps=6.0, rng=rng)
    specs: list[RequestSpec] = []
    for i in range(n):
        target_tokens = rng.randint(1024, 2048) if rng.random() < 0.75 else rng.randint(64, 128)
        max_tokens = rng.randint(64, 256)
        prompt = make_prompt(target_tokens, seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=max_tokens, send_delay_s=delays[i]))
    return specs


def _gen_prefix_caching(n: int, rng: random.Random) -> list[RequestSpec]:
    """Shared system prompt with unique suffixes."""
    delays = uniform_delays(n, rate_rps=8.0)
    specs: list[RequestSpec] = []
    for i in range(n):
        suffix_tokens = rng.randint(32, 128)
        prompt = make_shared_prefix_prompt(1024, suffix_tokens, suffix_seed=rng.randint(0, 2**31))
        specs.append(RequestSpec(prompt=prompt, max_tokens=256, send_delay_s=delays[i]))
    return specs


WORKLOADS: dict[str, WorkloadDef] = {
    "baseline": WorkloadDef(
        name="baseline",
        description="Sequential single requests (establish overhead floor)",
        generator=_gen_baseline,
        default_num_requests=10,
        default_warmup_requests=2,
        sequential=True,
    ),
    "continuous_batching": WorkloadDef(
        name="continuous_batching",
        description="Staggered arrivals, varying lengths (expose per-step admit/retire benefit)",
        generator=_gen_continuous_batching,
        default_num_requests=32,
        default_warmup_requests=2,
    ),
    "paged_attention": WorkloadDef(
        name="paged_attention",
        description="Bursty arrivals, bimodal lengths (expose memory waste from contiguous KV)",
        generator=_gen_paged_attention,
        default_num_requests=48,
        default_warmup_requests=2,
    ),
    "chunked_prefill": WorkloadDef(
        name="chunked_prefill",
        description="Long prompts, Poisson arrivals (expose ITL spikes from long prefills)",
        generator=_gen_chunked_prefill,
        default_num_requests=48,
        default_warmup_requests=2,
    ),
    "prefix_caching": WorkloadDef(
        name="prefix_caching",
        description="Shared system prompt (expose repeated prefill cost)",
        generator=_gen_prefix_caching,
        default_num_requests=48,
        default_warmup_requests=2,
    ),
}


# ---------------------------------------------------------------------------
# SSE client
# ---------------------------------------------------------------------------


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    spec: RequestSpec,
    request_idx: int,
) -> RequestResult:
    """Send a single request and parse the SSE stream."""
    result = RequestResult(
        request_idx=request_idx,
        prompt_tokens=len(spec.prompt) // CHARS_PER_TOKEN,
        completion_tokens=0,
        send_time_s=0.0,
        first_token_time_s=0.0,
        done_time_s=0.0,
    )

    body: dict[str, object] = {
        "model": model,
        "prompt": spec.prompt,
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
        "stream": True,
    }
    if spec.seed is not None:
        body["seed"] = spec.seed

    result.send_time_s = time.perf_counter()

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/completions", json=body, timeout=300.0
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                result.done_time_s = time.perf_counter()
                result.error = f"HTTP {resp.status_code}: {resp.text}"
                return result

            current_event = ""
            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                ts = time.perf_counter()
                data_str = line[len("data:") :].strip()

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if current_event == "token":
                    result.token_timestamps_s.append(ts)
                    if result.first_token_time_s <= 0:
                        result.first_token_time_s = ts

                elif current_event == "done":
                    result.done_time_s = ts
                    result.finish_reason = data.get("finish_reason", "")
                    usage = data.get("usage", {})
                    result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
                    result.completion_tokens = usage.get("completion_tokens", 0)

                elif current_event == "error":
                    result.done_time_s = ts
                    result.error = data.get("error", "unknown error")
                    return result

    except httpx.HTTPError as e:
        result.done_time_s = time.perf_counter()
        result.error = f"HTTP error: {e}"
    except Exception as e:
        result.done_time_s = time.perf_counter()
        result.error = f"Unexpected error: {e}"

    # Ensure done_time_s is set even if stream ended without a done event.
    if result.done_time_s <= 0:
        result.done_time_s = time.perf_counter()

    return result


# ---------------------------------------------------------------------------
# Report computation
# ---------------------------------------------------------------------------


def compute_report(
    workload_name: str,
    results: list[RequestResult],
    wall_clock_s: float,
) -> WorkloadReport:
    """Compute aggregate metrics from per-request results."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    total_output_tokens = sum(r.completion_tokens for r in successful)
    throughput = total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    ttft_values = [r.ttft_s * 1000 for r in successful if r.ttft_s > 0]
    all_itl = [v * 1000 for r in successful for v in r.itl_values_s]
    latency_values = [r.request_latency_s for r in successful if r.request_latency_s > 0]

    return WorkloadReport(
        workload_name=workload_name,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        wall_clock_s=wall_clock_s,
        throughput_output_tps=throughput,
        ttft_mean_ms=mean(ttft_values),
        ttft_p50_ms=percentile(ttft_values, 50),
        ttft_p95_ms=percentile(ttft_values, 95),
        ttft_p99_ms=percentile(ttft_values, 99),
        itl_mean_ms=mean(all_itl),
        itl_p50_ms=percentile(all_itl, 50),
        itl_p95_ms=percentile(all_itl, 95),
        itl_p99_ms=percentile(all_itl, 99),
        latency_mean_s=mean(latency_values),
        latency_p50_s=percentile(latency_values, 50),
        latency_p95_s=percentile(latency_values, 95),
        latency_p99_s=percentile(latency_values, 99),
        results=results,
    )


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


async def _send_with_delay(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    spec: RequestSpec,
    request_idx: int,
    start_time: float,
) -> RequestResult:
    """Wait until the scheduled send time, then send the request."""
    target = start_time + spec.send_delay_s
    now = time.perf_counter()
    if target > now:
        await asyncio.sleep(target - now)
    return await send_request(client, base_url, model, spec, request_idx)


async def run_workload(
    base_url: str,
    model: str,
    workload: WorkloadDef,
    num_requests: int | None,
    warmup_requests: int | None,
    seed: int,
) -> WorkloadReport:
    """Run a workload and return the report."""
    n = num_requests if num_requests is not None else workload.default_num_requests
    n_warmup = warmup_requests if warmup_requests is not None else workload.default_warmup_requests

    rng = random.Random(seed)
    total = n_warmup + n
    all_specs = workload.generator(total, rng)
    warmup_specs = all_specs[:n_warmup]
    measure_specs = all_specs[n_warmup:]

    async with httpx.AsyncClient() as client:
        # Warmup phase (sequential).
        if warmup_specs:
            print(f"  Warmup: sending {len(warmup_specs)} request(s) sequentially...")
            for i, spec in enumerate(warmup_specs):
                r = await send_request(client, base_url, model, spec, request_idx=-1)
                if r.error:
                    print(f"  Warmup request {i} error: {r.error}")

        # Measurement phase.
        print(f"  Measurement: sending {len(measure_specs)} request(s)...")
        wall_start = time.perf_counter()

        if workload.sequential:
            results: list[RequestResult] = []
            for i, spec in enumerate(measure_specs):
                r = await send_request(client, base_url, model, spec, request_idx=i)
                status = "ok" if r.error is None else f"error: {r.error}"
                print(f"    [{i + 1}/{n}] {r.completion_tokens} tokens, {status}")
                results.append(r)
        else:
            tasks = [
                asyncio.create_task(
                    _send_with_delay(client, base_url, model, spec, idx, wall_start)
                )
                for idx, spec in enumerate(measure_specs)
            ]
            results = list(await asyncio.gather(*tasks))

        wall_elapsed = time.perf_counter() - wall_start

    return compute_report(workload.name, results, wall_elapsed)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def health_check(base_url: str, model: str) -> None:
    """Send a minimal request to verify the server is reachable and model matches."""
    body: dict[str, object] = {
        "model": model,
        "prompt": "Hello",
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    try:
        async with (
            httpx.AsyncClient() as client,
            client.stream("POST", f"{base_url}/v1/completions", json=body, timeout=60.0) as resp,
        ):
            if resp.status_code == 422:
                await resp.aread()
                raise SystemExit(
                    f"Health check failed: model mismatch or invalid request.\n"
                    f"  Server returned 422: {resp.text}\n"
                    f"  Check that --model matches the server's loaded model."
                )
            if resp.status_code != 200:
                await resp.aread()
                raise SystemExit(
                    f"Health check failed: HTTP {resp.status_code}\n  Response: {resp.text}"
                )
            # Consume stream to completion.
            async for _ in resp.aiter_lines():
                pass
    except httpx.ConnectError as e:
        raise SystemExit(
            f"Health check failed: cannot connect to {base_url}\n"
            f"  Is the server running? Start with:\n"
            f"    uv run python -m infer.server --model {model}"
        ) from e
    except httpx.HTTPError as e:
        raise SystemExit(f"Health check failed: {e}") from e

    print(f"Health check passed: {base_url} serving {model}")


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_workload_report(report: WorkloadReport) -> None:
    """Print a single workload report table."""
    print()
    print(f"=== {report.workload_name} ===")
    print(f"Requests:   {report.successful_requests}/{report.total_requests} successful")
    if report.failed_requests > 0:
        print(f"Errors:     {report.failed_requests}")
    print(f"Wall clock: {report.wall_clock_s:.2f} s")
    print(f"Throughput: {report.throughput_output_tps:.1f} output tok/s")
    print()

    header = f"{'Metric':<20s}  {'Mean':>10s}  {'P50':>10s}  {'P95':>10s}  {'P99':>10s}"
    print(header)
    print("-" * len(header))

    print(
        f"{'TTFT (ms)':<20s}  {report.ttft_mean_ms:>10.1f}  {report.ttft_p50_ms:>10.1f}  "
        f"{report.ttft_p95_ms:>10.1f}  {report.ttft_p99_ms:>10.1f}"
    )
    print(
        f"{'ITL (ms)':<20s}  {report.itl_mean_ms:>10.1f}  {report.itl_p50_ms:>10.1f}  "
        f"{report.itl_p95_ms:>10.1f}  {report.itl_p99_ms:>10.1f}"
    )
    print(
        f"{'Latency (s)':<20s}  {report.latency_mean_s:>10.3f}  {report.latency_p50_s:>10.3f}  "
        f"{report.latency_p95_s:>10.3f}  {report.latency_p99_s:>10.3f}"
    )
    print()


def print_comparison_table(reports: list[WorkloadReport]) -> None:
    """Print a cross-workload comparison table."""
    print()
    print("=== Cross-Workload Comparison ===")
    print()

    header = (
        f"{'Workload':<24s}  {'OK':>4s}  {'Err':>4s}  {'Tput':>8s}  "
        f"{'TTFT P50':>9s}  {'TTFT P99':>9s}  "
        f"{'ITL P50':>8s}  {'ITL P99':>8s}  "
        f"{'Lat P50':>8s}  {'Lat P99':>8s}"
    )
    print(header)
    print("-" * len(header))

    for r in reports:
        print(
            f"{r.workload_name:<24s}  "
            f"{r.successful_requests:>4d}  {r.failed_requests:>4d}  "
            f"{r.throughput_output_tps:>7.1f}  "
            f"{r.ttft_p50_ms:>8.1f}  {r.ttft_p99_ms:>8.1f}  "
            f"{r.itl_p50_ms:>7.1f}  {r.itl_p99_ms:>7.1f}  "
            f"{r.latency_p50_s:>7.3f}  {r.latency_p99_s:>7.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def build_json_report(
    reports: list[WorkloadReport],
    *,
    server_url: str,
    model: str,
    seed: int,
) -> dict[str, object]:
    """Build a JSON-serializable report."""
    workloads_json: list[dict[str, object]] = []

    for report in reports:
        requests_json: list[dict[str, object]] = []
        for r in report.results:
            requests_json.append(
                {
                    "request_idx": r.request_idx,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "ttft_ms": round(r.ttft_s * 1000, 2),
                    "itl_values_ms": [round(v * 1000, 2) for v in r.itl_values_s],
                    "latency_s": round(r.request_latency_s, 4),
                    "finish_reason": r.finish_reason,
                    "error": r.error,
                }
            )

        workloads_json.append(
            {
                "workload_name": report.workload_name,
                "total_requests": report.total_requests,
                "successful_requests": report.successful_requests,
                "failed_requests": report.failed_requests,
                "wall_clock_s": round(report.wall_clock_s, 3),
                "throughput_output_tps": round(report.throughput_output_tps, 1),
                "ttft_mean_ms": round(report.ttft_mean_ms, 2),
                "ttft_p50_ms": round(report.ttft_p50_ms, 2),
                "ttft_p95_ms": round(report.ttft_p95_ms, 2),
                "ttft_p99_ms": round(report.ttft_p99_ms, 2),
                "itl_mean_ms": round(report.itl_mean_ms, 2),
                "itl_p50_ms": round(report.itl_p50_ms, 2),
                "itl_p95_ms": round(report.itl_p95_ms, 2),
                "itl_p99_ms": round(report.itl_p99_ms, 2),
                "latency_mean_s": round(report.latency_mean_s, 4),
                "latency_p50_s": round(report.latency_p50_s, 4),
                "latency_p95_s": round(report.latency_p95_s, 4),
                "latency_p99_s": round(report.latency_p99_s, 4),
                "requests": requests_json,
            }
        )

    return {
        "benchmark": "serving",
        "server_url": server_url,
        "model": model,
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "workloads": workloads_json,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    workload_choices = [*WORKLOADS.keys(), "all"]

    parser = argparse.ArgumentParser(
        description="Serving benchmark: measure throughput, TTFT, and ITL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "workloads:\n"
            "  baseline              Sequential single requests (overhead floor)\n"
            "  continuous_batching   Staggered arrivals, varying lengths\n"
            "  paged_attention       Bursty arrivals, bimodal lengths\n"
            "  chunked_prefill       Long prompts, Poisson arrivals\n"
            "  prefix_caching        Shared system prompt\n"
            "  all                   Run all workloads sequentially\n"
        ),
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (must match server's loaded model)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=workload_choices,
        default="all",
        help="Workload to run (default: all)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Override default number of requests per workload",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help="Override default warmup requests per workload",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON report output path (default: auto-generated)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving JSON report",
    )

    args = parser.parse_args()
    asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> None:
    """Async main."""
    base_url = args.server.rstrip("/")

    # Health check.
    await health_check(base_url, args.model)

    # Select workloads.
    workload_names = list(WORKLOADS.keys()) if args.workload == "all" else [args.workload]

    # Run workloads.
    reports: list[WorkloadReport] = []
    for name in workload_names:
        workload = WORKLOADS[name]
        n = args.num_requests if args.num_requests is not None else workload.default_num_requests
        print(f"\n--- Running workload: {name} ({n} requests) ---")
        print(f"  {workload.description}")
        report = await run_workload(
            base_url,
            args.model,
            workload,
            num_requests=args.num_requests,
            warmup_requests=args.warmup_requests,
            seed=args.seed,
        )
        reports.append(report)
        print_workload_report(report)

    # Cross-workload comparison.
    if len(reports) > 1:
        print_comparison_table(reports)

    # Save JSON report.
    if not args.no_save:
        if args.output:
            report_path = Path(args.output)
        else:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            model_slug = args.model.replace("/", "_")
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"serving_{model_slug}_{timestamp}.json"

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_json = build_json_report(
            reports,
            server_url=base_url,
            model=args.model,
            seed=args.seed,
        )
        with open(report_path, "w") as f:
            json.dump(report_json, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
