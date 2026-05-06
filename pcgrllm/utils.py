import os

def create_message_box(text):
    lines = text.split('\n')

    max_length = max(len(line) for line in lines)

    box_length = max_length + 6

    top_bottom_border = "#" * box_length
    middle_lines = [f"### {line.ljust(max_length)} ###" for line in lines]

    return f"{top_bottom_border}\n" + "\n".join(middle_lines) + f"\n{top_bottom_border}"

def get_textfile_tail(log_path, tail: int = 60) -> str:
    with open(log_path, 'rb') as f:
        # Move to the end of the file
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        block_size = 1024
        data = []
        while file_size > 0 and len(data) < tail:
            if file_size - block_size > 0:
                f.seek(-block_size, os.SEEK_CUR)
            else:
                f.seek(0)
                block_size = file_size
            chunk = f.read(block_size).splitlines()
            data = chunk + data
            file_size -= block_size
            f.seek(file_size, os.SEEK_SET)

        # Trim the list to the last 'tail' lines
        if len(data) > tail:
            data = data[-tail:]
        logs = [line.decode('utf-8') for line in data]

    return '\n'.join(logs)

