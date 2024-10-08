# 定义颜色代码
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

# 示例打印函数
def colored_print(color, text):
    print(f"{color}{text}{Colors.RESET}")

# # 使用示例
# colored_print(Colors.RED, "This is red text")
# colored_print(Colors.GREEN, "This is green text")
# colored_print(Colors.BLUE, "This is blue text")
# colored_print(Colors.MAGENTA, "This is magenta text")