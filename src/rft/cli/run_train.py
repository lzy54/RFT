from __future__ import annotations

from pathlib import Path
import typer

from rft.train.llamafactory import run_llamafactory_train

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/llamafactory_sft.yaml"),
        help="Path to LLaMA Factory SFT config",
    ),
    env_name: str = typer.Option(
        "rft-train",
        help="Conda env name for LLaMA Factory",
    ),
):
    if not config.exists():
        raise FileNotFoundError(config)

    run_llamafactory_train(
        config_path=config,
        env_name=env_name,
    )

    typer.echo("[OK] LLaMA Factory training finished.")


if __name__ == "__main__":
    app()
