from src.muvis_align.util import metric_to_rgb


def colors_test():
    steps = 100
    for i in range(steps):
        value = i / steps
        print(value, metric_to_rgb(value))

if __name__ == '__main__':
    colors_test()
