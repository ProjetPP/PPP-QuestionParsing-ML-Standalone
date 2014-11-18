import unittest


def main(): # pragma: no cover
    test_suite = unittest.TestLoader().discover('tests/')
    results = unittest.TextTestRunner(verbosity=1).run(test_suite)
    if results.errors or results.failures:
        exit(1)
    else:
        exit(0)

if __name__ == '__main__':
    main()
