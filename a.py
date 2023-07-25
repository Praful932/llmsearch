from llmsearch.utils.mem_utils import get_traceback


def foo():
    print(get_traceback())


foo()
