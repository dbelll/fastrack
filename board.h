/*
 *  board.h
 *  fastrack
 *
 *  Created by Dwight Bell on 1/1/11.
 *  Copyright 2011 dbelll. All rights reserved.
 *

	Helpers for using the game board
 
 */


// calculate index number for given row and column
unsigned index4rc(unsigned row, unsigned col)
{
	return row * g_p.board_width + col;
}

// generate string for the cell at row and column
char *move_string(char *buff, unsigned col, unsigned row)
{
	buff[0] = 'a' + col;
	buff[1] = '1' + row;
	return buff;
}

// generate string for the cell at given index
char *move_stringi(char *buff, unsigned i)
{
	return move_string(buff, i % g_p.board_width, i / g_p.board_width);
}

unsigned is_empty(unsigned *board)
{
	for (int i = 0; i < g_p.board_size; i++) {
		if (board[i]) return 0;
	}
	return 1;
}

unsigned not_empty(unsigned *board){ return !is_empty(board); }

unsigned game_won(float reward){
	return reward > 0.5f;
}

char char_for_index(unsigned i, unsigned *state)
{
	unsigned s0 = X_BOARD(state)[i];
	unsigned s1 = O_BOARD(state)[i];
	if (s0 && s1) return '?';
	else if (s0) return 'X';
	else if (s1) return 'O';
	return '.';
}

char char_for_board(unsigned row, unsigned col, unsigned *board)
{
	unsigned index = index4rc(row, col);
	//	printf("index for row %d, col%d is %d\n", row, col, index);
	//	printf("value of board there is %d\n", board[index]);
	char ch = board[index] ? 'X' : '.';
	//	printf("char_for_board row=%d, col=%d is %c\n", row, col, ch);
	return ch;
}

char char_for_cell(unsigned row, unsigned col, unsigned *state)
{
	return char_for_index(index4rc(row, col), state);
}


