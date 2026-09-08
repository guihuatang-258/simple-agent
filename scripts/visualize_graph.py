"""Export the customer-service LangGraph structure for debugging."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from agent import build_agent


class _StructureOnlyModel:
    """Placeholder that keeps graph visualization independent of model config."""

    async def ainvoke(self, *_args, **_kwargs):
        raise RuntimeError("可视化脚本不会执行模型。")


def build_graph_structure():
    """Build the compiled graph without loading tools or a real LLM."""

    return build_agent(model=_StructureOnlyModel()).get_graph()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="输出客服 LangGraph 的静态结构图。")
    parser.add_argument(
        "--format",
        choices=("png", "mermaid", "ascii"),
        default="png",
        help="输出格式，默认生成 PNG。",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="输出文件路径；PNG 默认写入 customer-service-graph.png。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    graph = build_graph_structure()

    if args.format == "mermaid":
        content = graph.draw_mermaid()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
            print(f"Mermaid 图已写入：{args.output.resolve()}")
        else:
            print(content)
        return 0

    if args.format == "ascii":
        try:
            print(graph.draw_ascii())
        except ImportError as exc:
            raise SystemExit(
                "ASCII 输出需要 grandalf，请先执行：pip install grandalf"
            ) from exc
        return 0

    output = args.output or Path("customer-service-graph.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        graph.draw_mermaid_png(output_file_path=str(output))
    except Exception as exc:
        raise SystemExit(
            f"PNG 渲染失败：{exc}\n"
            "默认 PNG 渲染需要访问 Mermaid 服务；可改用 --format mermaid。"
        ) from exc
    print(f"PNG 图已生成：{output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
