import numpy as np


def read_csv(
    filepath,
    delimiter=',',
    missing_values=None,
    filling_values=np.nan,
    dtype=None,
    skip_header=0
):
    if missing_values is None:
        missing_values = ["", "NA", "N/A", "null", "None"]

    try:
        data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            missing_values=missing_values,
            filling_values=filling_values,
            dtype=dtype,
            ndmin=2,
            skip_header=skip_header
        )
    except (OSError, ValueError) as e:
        raise ValueError(
            f"Failed to load CSV file '{filepath}': {e}"
        ) from e

    data = np.array(data)

    return data


def read_csv_chunks(
    filepath,
    delimiter=',',
    chunk_size=None,
    missing_values=None,
    filling_values=np.nan,
    dtype=None,
    skip_header=0
):
    if chunk_size is None or chunk_size <= 0:
        raise ValueError("Chunk size must be > 0")

    if missing_values is None:
        missing_values = ["", "NA", "N/A", "null", "None"]

    with open(filepath, "r") as f:
        for _ in range(skip_header):
            next(f, None)

        while True:
            lines = []

            # Read up to chunk_size lines
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            if not lines:
                break

            # Convert the current chunk to NumPy array
            data = np.genfromtxt(
                lines,
                delimiter=delimiter,
                missing_values=missing_values,
                filling_values=filling_values,
                dtype=dtype,
                ndmin=2
            )

            data = np.array(data)

            yield data