import pygame

# Initial Sudoku board
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
]

# Display setup
WIDTH, HEIGHT = 550, 550
CELL_SIZE = 50
OFFSET = 50
BUFFER = 7

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SUDOKU")
FONT = pygame.font.SysFont("Palatino", 30)

def find_empty_sqr(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def validate_our_value(board, val, coord):
    row, col = coord

    # Check row
    if val in board[row]:
        return False

    # Check column
    for i in range(9):
        if board[i][col] == val:
            return False

    # Check 3x3 cell
    box_x = col // 3 * 3
    box_y = row // 3 * 3
    for i in range(box_y, box_y + 3):
        for j in range(box_x, box_x + 3):
            if board[i][j] == val:
                return False

    return True

def sudoku_solver(win):
    find = find_empty_sqr(s_board)
    if not find:
        return True

    row, col = find

    for k in range(1, 10):
        if validate_our_value(s_board, k, (row, col)):
            s_board[row][col] = k

            pygame.draw.rect(win, WHITE, ((col + 1) * CELL_SIZE + BUFFER, (row + 1) * CELL_SIZE + BUFFER, CELL_SIZE - 2 * BUFFER, CELL_SIZE - 2 * BUFFER))
            sol = FONT.render(str(k), True, RED)
            win.blit(sol, ((col + 1) * CELL_SIZE + 15, (row + 1) * CELL_SIZE + 15))
            pygame.display.update()
            pygame.time.delay(30)

            if sudoku_solver(win):
                return True

            s_board[row][col] = 0
            pygame.draw.rect(win, WHITE, ((col + 1) * CELL_SIZE + BUFFER, (row + 1) * CELL_SIZE + BUFFER, CELL_SIZE - 2 * BUFFER, CELL_SIZE - 2 * BUFFER))
            pygame.display.update()

    return False

def draw_window():
    WIN.fill(WHITE)

    # Draw grid lines
    for i in range(10):
        line_width = 4 if i % 3 == 0 else 2
        pygame.draw.line(WIN, BLACK, (OFFSET + i * CELL_SIZE, OFFSET), (OFFSET + i * CELL_SIZE, OFFSET + 9 * CELL_SIZE), line_width)
        pygame.draw.line(WIN, BLACK, (OFFSET, OFFSET + i * CELL_SIZE), (OFFSET + 9 * CELL_SIZE, OFFSET + i * CELL_SIZE), line_width)

    # Draw board numbers
    for i in range(9):
        for j in range(9):
            num = s_board[i][j]
            if num != 0:
                val_text = FONT.render(str(num), True, BLUE)
                WIN.blit(val_text, ((j + 1) * CELL_SIZE + 15, (i + 1) * CELL_SIZE + 15))

    pygame.display.update()
    sudoku_solver(WIN)

def main():
    clock = pygame.time.Clock()
    run = True
    draw_window()
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    pygame.quit()

if __name__ == "__main__":
    main()
