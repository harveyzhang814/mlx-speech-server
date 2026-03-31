import click
from app.config import ServerConfig
from app.server import run


@click.command()
@click.option("--host", default=None, help="Bind host (default: 0.0.0.0)")
@click.option("--port", default=None, type=int, help="Bind port (default: 8000)")
@click.option("--model-path", default=None, help="HuggingFace repo or local path for the Whisper model")
@click.option("--quantize", default=None, type=click.Choice(["4", "8"]), help="Quantization bits (use a pre-quantized model path)")
@click.option("--queue-max-size", default=None, type=int, help="Max concurrent+waiting requests (default: 10)")
@click.option("--queue-timeout", default=None, type=float, help="Queue wait timeout in seconds (default: 300)")
@click.option("--memory-cleanup-interval", default=None, type=int, help="Clear Metal cache every N requests (default: 20)")
@click.option("--log-level", default=None, type=click.Choice(["debug", "info", "warning", "error"]))
def cli(host, port, model_path, quantize, queue_max_size, queue_timeout, memory_cleanup_interval, log_level):
    """mlx-speech-server: OpenAI-compatible Whisper API on Apple Silicon."""
    config = ServerConfig.from_env()

    # CLI args override env vars
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if model_path is not None:
        config.model_path = model_path
    if quantize is not None:
        config.quantize = int(quantize)
    if queue_max_size is not None:
        config.queue_max_size = queue_max_size
    if queue_timeout is not None:
        config.queue_timeout = queue_timeout
    if memory_cleanup_interval is not None:
        config.memory_cleanup_interval = memory_cleanup_interval
    if log_level is not None:
        config.log_level = log_level

    run(config)


if __name__ == "__main__":
    cli()
