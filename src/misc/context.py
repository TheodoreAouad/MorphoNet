"""Context usefull classes mainly to log training details."""

from typing import Optional, TextIO, Any
import sys
import logging
import tempfile
import mlflow


class OutputManagment:
    """Class allowing to copy standard output into a file."""

    class Tee:
        """Class mimicking the tee command from Unix."""

        def __init__(
            self, output_managment: "OutputManagment", std: TextIO
        ) -> None:
            self.std = std
            self.output_managment = output_managment

        def write(self, text: str) -> int:
            """Writing output to stdout and a file if necessary."""
            out = self.std.write(text)
            if self.output_managment.write_to_file:
                out &= self.output_managment.std_fd.write(text)

            return out

        def flush(self) -> None:
            """Flushing output(s)."""
            self.std.flush()
            if self.output_managment.write_to_file:
                self.output_managment.std_fd.flush()

    std_fd: TextIO
    write_to_file: bool

    def __init__(self) -> None:
        self.std_path: Optional[str] = None

        self.stdout = OutputManagment.Tee(self, sys.stdout)
        self.stderr = OutputManagment.Tee(self, sys.stderr)

        sys.stdout = self.stdout  # type: ignore
        sys.stderr = self.stderr  # type: ignore

    def set(self, write_to_file: bool = True) -> None:
        """Disabe or enable and prepare file writing."""
        self.write_to_file = write_to_file

        if self.write_to_file:
            _, self.std_path = tempfile.mkstemp(suffix=".txt")
            self.std_fd = open(  # pylint: disable=consider-using-with
                self.std_path, "w", encoding="utf-8"
            )

    def __del__(self) -> None:
        if self.std_fd is not None:
            self.std_fd.close()


class Task:
    """Context manager logging when a task is done."""

    def __init__(self, message: str):
        self.message = message

    def __enter__(self) -> None:
        logging.info("%s...", self.message)

    def __exit__(self, *args: Any) -> None:
        logging.info("Done")


class RunContext:
    """Context allowing logging output as an artifact to mlflow."""

    def __init__(
        self, run: mlflow.ActiveRun, output_managment: OutputManagment
    ):
        self.run = run
        self.output_managment = output_managment

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> None:
        print(self.run.info.run_id)

        sys.stdout.flush()
        sys.stderr.flush()

        mlflow.log_artifact(self.output_managment.std_path, "logs")
