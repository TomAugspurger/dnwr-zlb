def strip_header(f, n=10):
    for i in range(n):
        f.readline()


def pull_item(line, i):
    """
    If you do have a match, use this to pull the data name, width,
    and column locations out.
    """
    parts = line.split('\t')
    try:
        id, width, name = parts[0:6:2]
        location = parts[-1].strip()
        return id, width, name, location
    except ValueError:
        return i, line


def item_identifier(line, i):
    """Determines if a line is an actual record."""
    if not line.startswith(('\t', '\r', 'NOTE', '(4', ' ')):
        return True
    else:
        return False


def main(file_):
    strip_header(file_)
    xs = []
    warnings = []
    for i, line in enumerate(file_):
        if item_identifier(line, i):
            ret = pull_item(line, i)
            try:
                xs.append(','.join(ret))
            except TypeError:
                warnings.append(ret)
    return xs, warnings

if __name__ == '__main__':
    import sys
    try:
        in_file, out_file = sys.argv[1:]
    except IndexError:
        out_file = 'parsed' + in_file
    xs = main(in_file)
    with open(out_file, 'wt') as f:
        f.write(xs)
