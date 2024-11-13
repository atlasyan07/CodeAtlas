import pygame

s_board = [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ]  # The hardest Sudoku puzzle in the world

WIDTH, HEIGTH = 550, 550
WIN = pygame.display.set_mode((WIDTH, HEIGTH))
pygame.display.set_caption("SUDOKU")
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
FPS = 60
BUFFER = 7

def find_empty_sqr(bo):
    for i in range(len(bo[0])):
        for j in range(len(bo[1])):
            if bo[i][j] == " ":
                return (i, j)
    return None  # A function that finds the empty square within a grid

def validate_our_value(bo, val, cord):  # bo is the board, val is number we want evaluated and cord is the cordinate of our desired square
    # check row
    for i in range(len(bo[0])):
        if bo[cord[0]][i] == val and cord[1] != i:
            return False

    # check column
    for j in range(len(bo[1])):
        if bo[j][cord[1]] == val and cord[0] != j:
            return False

    # check cell
    cell_x = cord[1] // 3
    cell_y = cord[0] // 3
    for i in range(cell_y * 3, cell_y * 3 + 3):
        for j in range(cell_x * 3, cell_x * 3 + 3):
            if bo[i][j] == val and cord != (i, j):
                return False
    return True

def sudoku_solver(WIN):
    pygame.init()
    myfont = pygame.font.SysFont("Palatino", 30)
    find = find_empty_sqr(s_board)
    if not find:
        return True
    else:
        row, col = find

    for k in range(1, 10):
        if validate_our_value(s_board, k, (row, col)):
            s_board[row][col] = k
            pygame.draw.rect(WIN, WHITE, ((col + 1) * 50 + BUFFER, (row + 1) * 50 + 7, 50 - 10, 50 - 10))
            SOL = myfont.render(str(k), True, RED)
            WIN.blit(SOL, ((col + 1) * 50 + 15, (row + 1) * 50 + 15))
            pygame.display.update()

            if sudoku_solver(WIN):
                return True
            else:
                s_board[row][col] = " "
                pygame.display.update()

    return False

def draw_window():
    pygame.init()
    WIN.fill(WHITE)
    myfont = pygame.font.SysFont("Palatino", 30)
    for i in range(10):
        if i % 3 == 0:
            pygame.draw.line(WIN, BLACK, (50 + 50 * i, 50), (50 + 50 * i, 500), 4)
            pygame.draw.line(WIN, BLACK, (50, 50 + 50 * i), (500, 50 + 50 * i), 4)
        else:
            pygame.draw.line(WIN, BLACK, (50 + 50 * i, 50), (50 + 50 * i, 500), 2)
            pygame.draw.line(WIN, BLACK, (50, 50 + 50 * i), (500, 50 + 50 * i), 2)

    for i in range(len(s_board[0])):
        for j in range(len(s_board[1])):
            if s_board[i][j] == 0:
                s_board[i][j] = " "

            if j == 8:
                value1 = myfont.render(str(s_board[i][j]), True, BLUE)
                WIN.blit(value1, ((j + 1) * 50 + 15, (i + 1) * 50 + 15))
            else:
                value2 = myfont.render(str(s_board[i][j]), True, BLUE)
                WIN.blit(value2, ((j + 1) * 50 + 15, (i + 1) * 50 + 15))

    pygame.display.update()
    sudoku_solver(WIN)

def main():
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():  # gives us the list of all events happening in pygame.
            if event.type == pygame.QUIT:
                run = False

        draw_window()

    pygame.quit()

if __name__ == '__main__':
    main()
