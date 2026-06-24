import argparse
import inspect
import os
import uuid
from typing import Dict, List, Tuple

import gradio as gr
import torch
from melo.api import TTS

from openvoice import se_extractor
from openvoice.api import ToneColorConverter


CKPT_CONVERTER = "checkpoints_v2/converter"
BASE_SE_DIR = "checkpoints_v2/base_speakers/ses"
OUTPUT_DIR = "outputs"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lazily initialize TTS models to reduce startup time and memory usage.
TTS_MODELS: Dict[str, TTS] = {}
SPEAKER_CACHE: Dict[str, Dict[str, Dict[str, str]]] = {}


def load_tone_color_converter() -> ToneColorConverter:
    converter = ToneColorConverter(f"{CKPT_CONVERTER}/config.json", device=DEVICE)
    converter.load_ckpt(f"{CKPT_CONVERTER}/checkpoint.pth")
    return converter


tone_color_converter = load_tone_color_converter()


def patch_starlette_template_response_compat() -> None:
    """Allow Gradio 3 old TemplateResponse call style on newer Starlette."""
    try:
        from starlette.templating import Jinja2Templates
    except Exception:
        return

    original = Jinja2Templates.TemplateResponse
    try:
        params = list(inspect.signature(original).parameters.keys())
    except Exception:
        return

    # New Starlette signature is: (self, request, name, context, ...)
    if len(params) < 3 or params[1] != "request" or params[2] != "name":
        return

    def _compat(self, *args, **kwargs):
        # Gradio 3 style: TemplateResponse(name, context)
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = context.get("request")
            if request is None:
                raise TypeError("Missing request in template context for compatibility mode")
            return original(self, request, name, context, *args[2:], **kwargs)

        return original(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _compat


patch_starlette_template_response_compat()


SUPPORTED_LANGUAGES = ["EN", "ZH", "JP"]
DEFAULT_TEXTS = {
    "EN": "Did you ever hear a folk tale about a giant turtle?",
    "ZH": "在这次vacation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。",
    "JP": "彼は毎朝ジョギングをして体を健康に保っています。"
}


# Map model speaker key -> metadata used in UI and conversion.
def _build_speaker_meta(language: str) -> Dict[str, Dict[str, str]]:
    model = get_tts(language)
    speaker_ids = model.hps.data.spk2id
    result: Dict[str, Dict[str, str]] = {}

    for raw_key, speaker_id in speaker_ids.items():
        normalized = raw_key.lower().replace("_", "-")
        candidates = [
            os.path.join(BASE_SE_DIR, f"{normalized}.pth"),
            os.path.join(BASE_SE_DIR, f"{raw_key}.pth"),
            os.path.join(BASE_SE_DIR, f"{str(raw_key).lower()}.pth"),
        ]
        se_path = next((p for p in candidates if os.path.exists(p)), None)
        if se_path is None:
            continue

        result[str(raw_key)] = {
            "speaker_id": str(speaker_id),
            "se_path": se_path,
        }

    return result


def get_tts(language: str) -> TTS:
    if language not in TTS_MODELS:
        TTS_MODELS[language] = TTS(language=language, device=DEVICE)
    return TTS_MODELS[language]


def get_speakers(language: str) -> List[str]:
    if language not in SPEAKER_CACHE:
        SPEAKER_CACHE[language] = _build_speaker_meta(language)
    return list(SPEAKER_CACHE[language].keys())


def update_speaker_choices(language: str):
    speakers = get_speakers(language)
    default_speaker = speakers[0] if speakers else None
    default_text = DEFAULT_TEXTS.get(language, "")
    return gr.update(choices=speakers, value=default_speaker), default_text


def run_clone(
    text: str,
    language: str,
    speaker_key: str,
    speed: float,
    reference_audio: str,
) -> Tuple[str, str]:
    if not reference_audio or not os.path.exists(reference_audio):
        raise gr.Error("请先上传参考音频文件。")

    text = (text or "").strip()
    if len(text) < 2:
        raise gr.Error("请输入至少 2 个字符的文本。")

    if language not in SUPPORTED_LANGUAGES:
        raise gr.Error(f"不支持的语言: {language}")

    if language not in SPEAKER_CACHE:
        SPEAKER_CACHE[language] = _build_speaker_meta(language)

    speaker_meta = SPEAKER_CACHE[language].get(speaker_key)
    if speaker_meta is None:
        raise gr.Error("当前语言下无可用 base speaker，请切换语言或检查 checkpoints_v2/base_speakers/ses。")

    model = get_tts(language)

    target_se, _ = se_extractor.get_se(reference_audio, tone_color_converter, vad=True)

    source_se = torch.load(speaker_meta["se_path"], map_location=DEVICE)

    if torch.backends.mps.is_available() and DEVICE == "cpu":
        torch.backends.mps.is_available = lambda: False

    req_id = uuid.uuid4().hex[:8]
    src_path = os.path.join(OUTPUT_DIR, f"tmp_{req_id}.wav")
    out_path = os.path.join(OUTPUT_DIR, f"clone_{language}_{speaker_key}_{req_id}.wav")

    model.tts_to_file(text, int(speaker_meta["speaker_id"]), src_path, speed=float(speed))

    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=out_path,
        message="@MyShell",
    )

    info = (
        f"完成: language={language}, speaker={speaker_key}, speed={speed}, device={DEVICE}.\n"
        f"输出文件: {out_path}"
    )
    return out_path, info


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OpenVoice V2 MeloTTS Clone Demo") as demo:
        gr.Markdown("## OpenVoice V2 + MeloTTS 语音克隆")
        gr.Markdown(
            "上传参考音频后，先由 MeloTTS 生成基础语音，再通过 ToneColorConverter 转换到目标音色。"
        )

        with gr.Row():
            with gr.Column():
                language = gr.Dropdown(
                    choices=SUPPORTED_LANGUAGES,
                    value="ZH",
                    label="语言 / Language",
                )
                speaker = gr.Dropdown(label="Base Speaker")
                text = gr.Textbox(
                    label="合成文本 / Text",
                    value=DEFAULT_TEXTS["ZH"],
                    lines=4,
                )
                speed = gr.Slider(
                    minimum=0.7,
                    maximum=1.3,
                    value=1.0,
                    step=0.05,
                    label="语速 / Speed",
                )
                reference_audio = gr.Audio(
                    source="upload",
                    type="filepath",
                    label="参考音频（要克隆的音色，gradio 3.48 仅支持 upload 或 microphone 二选一）",
                )
                run_button = gr.Button("开始克隆", variant="primary")

            with gr.Column():
                out_audio = gr.Audio(label="克隆结果", type="filepath", autoplay=True)
                info = gr.Textbox(label="状态", lines=4)

        language.change(
            fn=update_speaker_choices,
            inputs=[language],
            outputs=[speaker, text],
        )

        run_button.click(
            fn=run_clone,
            inputs=[text, language, speaker, speed, reference_audio],
            outputs=[out_audio, info],
        )

        # Initialize speakers for default language when app starts.
        demo.load(
            fn=update_speaker_choices,
            inputs=[language],
            outputs=[speaker, text],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="Expose public Gradio URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()

    demo = build_demo()
    try:
        demo.queue().launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    except ValueError as exc:
        msg = str(exc)
        if "shareable link must be created" not in msg:
            raise
        # Some remote/proxy environments cannot access localhost unless share mode is enabled.
        demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=True)


if __name__ == "__main__":
    main()
