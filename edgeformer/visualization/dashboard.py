def print_report(report: dict):
    width = 52
    print("=" * width)
    print("EdgeFormer Hardware Report".center(width))
    print("=" * width)
    for key, value in report.items():
        print(f"{key:28s} {value}")
    print("=" * width)
