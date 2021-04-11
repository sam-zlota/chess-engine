#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io, chess,chess.pgn
import pandas as pd
import random
import time
import multiprocessing, os
from IPython.display import SVG, display
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# ## Constants

# In[2]:


BLACK = False 
WHITE = True

NUM_PROCESSES = 4
dark_diagonals = list(range(7,58,7)) # [8, 15, 22, 29, 36, 43, 50, 57]
light_diagonals = list(range(0,65,9)) # [1, 10, 19, 28, 37, 46, 55, 64]
both_diagonals = dark_diagonals + light_diagonals
center_squares = [27, 28, 35, 36]

piece_vals = {chess.PAWN:1, chess.BISHOP:3, chess.KNIGHT:3, chess.ROOK:5, chess.QUEEN:9}


# In[3]:




mg_value = (82, 337, 365, 477, 1025,  0)

mg_pawn_table = (
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
)


mg_knight_table = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
)

mg_bishop_table = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)

mg_rook_table = (
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
)

mg_queen_table = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)

mg_king_table = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)


# ## Simple Evaluation Functions

# In[4]:


class EvalFunc:
    """
    each function has a name which corresponds to its row in the dataset and a function
    """
    def apply(self, board,player):
        return None

class MatSumEvalFunc(EvalFunc):
    """
        Calculates Material Sum
    """
    name = "material sum"
    def apply(self, board, player):
        mat_sum = 0
        for piece_type in list(piece_vals.keys()):
            mat_sum += (len(board.pieces(piece_type, player)) * piece_vals[piece_type])
        return mat_sum

class PieceSquaresEvalFunc(EvalFunc):
    """
       Evaluates board based on positions of pieces, using the above tables
    """
    name = "piece squares sum"
    def apply(self, board,player):
        piece_map = board.piece_map()
        result = 0
        for square in list(piece_map.keys()):
            if piece_map[square].color == player :
                piece = piece_map[square]
                #print('old posn: ', square)
                square_file = square % 8

                square_rank = 7 - (square // 8)

                square_posn = 8 * square_rank + square_file
                
                if player is False:
                    square_posn = square_posn^56
                    
                
                #pawn
                if piece.piece_type is chess.PAWN:
                    result += (mg_pawn_table[square_posn] + mg_value[0])
                    
                #bishop
                elif piece.piece_type is chess.BISHOP:
                    result += (mg_bishop_table[square_posn] + mg_value[2])
                
                #knight
                elif piece.piece_type is chess.KNIGHT:
                    result += (mg_knight_table[square_posn] + mg_value[1])
                
                #queen
                elif piece.piece_type is chess.QUEEN:
                    result += (mg_queen_table[square_posn] + mg_value[4])
                
                #king
                elif piece.piece_type is chess.KING:
                    result += mg_king_table[square_posn]
                    
                #rook
                elif piece.piece_type is chess.ROOK:
                    result += (mg_rook_table[square_posn] + mg_value[3])
        return result
class DiagonalControlEvalFunc(EvalFunc):
    """
        Evaluates board based on how many diagonals are controlled by players queens and bishops
    """
    name = "diagonal control"
    def apply(self, board,player): 
        amount = 0
        for pos, piece in board.piece_map().items():
            if piece.color == player and pos in both_diagonals:
                if piece.piece_type is chess.QUEEN or piece.piece_type is chess.BISHOP:
                    amount+=1
        return amount
    
class CenterControlEvalFunc(EvalFunc):
    """
        Evaluates board based on how much of the center is controlled
    """
    name = "center control"
    def apply(self, board,player):
        #4 center squares
        #attacking center squares
        amount = 0
        piece_map = board.piece_map()
        for csquare in center_squares:
            if csquare in list(piece_map.keys()) and piece_map[csquare].color == player:
                amount+=2
            if board.is_attacked_by(player,csquare):
                amount+=1
        return amount
    
class PawnControlEvalFunc(EvalFunc):
    """
        TODO??? pawn islands?
    """
    name = "pawn control"
    def apply(self,board,player):
        piece_map = board.piece_map()
        player = not player
        result = 0
        file_dict = dict()
        for square in list(piece_map.keys()):
            if piece_map[square].color == player and piece_map[square].piece_type == 1:
                file_dict[chess.square_file(square)] = 1 

        isPawn = False
        for file in range(8) :
            #Add an island
            if isPawn and file_dict.get(file,0) == 0 :
                result+=1
                isPawn = False
            if file_dict.get(file,0) == 1 :
                isPawn = True

        #add the last pawn island
        if isPawn :
            result +=1
        return result
    
class DoublePawnsEvalFunc(EvalFunc):
    """
        Evaluates board based on how many pawns are doubled by the other player
    """
    name = "doubled pawns"
    def apply(self,board,player):
        piece_map = board.piece_map()
        player = not player
        result = 0
        file_dict = dict()
        for square in list(piece_map.keys()):
            if piece_map[square].color == player and piece_map[square].piece_type == 1:
                if file_dict.get(chess.square_file(square),0) == 1 :
                    result+=1
                else :
                    file_dict[chess.square_file(square)] = 1 
        return result
            

class MobilityEvalFunc(EvalFunc):
    """
        Evaluates board based on how many moves the player has
    """
    name = "mobility"
    def apply(self,board,player):
        return len(list(board.legal_moves))

class BothBishopsEvalFunc(EvalFunc):
    """
        Evaluates board based on whether or not both bishops are still on board.
    """
    name = "both bishops"
    def apply(self,board,player):
        return int(len(board.pieces(3,player)) == 2)
    
class NotAllPawnsEvalFunc(EvalFunc):
    """
        Evaluates board based on how many pawns there are. Less than 8 is better.
    """
    name = "not all pawns"
    def apply(self, board,player):
        return int(len(board.pieces(1,player)) < 8)

class PinEvalFunc(EvalFunc):
    """
        Evaluates board based on how many and of what type pieces are pinned for the other player
    """
    name = "pinned evaluation"
    def apply(self, board,player):
        player = not player
        piece_map = board.piece_map()
        result = 0
        for square in list(piece_map.keys()):
            if piece_map[square].color == player and board.is_pinned(player, square):
                result += piece_vals.get(piece_map[square].piece_type, 0)
        return result
    
class AttackerEvalFunc(EvalFunc):
    """
        Evaluates board based on how many and of what type pieces this player is attacking.
    """
    name = "attacker evalutation"
    def apply(self, board,player):
        player_other = not player
        piece_map = board.piece_map()
        result = 0
        for square in list(piece_map.keys()):
            if piece_map[square].color == player_other and board.is_attacked_by(player, square):
                
                result += piece_vals.get(piece_map[square].piece_type, 0)
                
        return result

class HasQueenEvalFunc(EvalFunc):
    """
        Evaluates board based on whether or not the player has their queen or not.
    """
    name = "has queen"
    def apply(self, board, player):
        return len(board.pieces(chess.QUEEN,player))

EVAL_FUNC_LIST = [MatSumEvalFunc(), PawnControlEvalFunc(), PieceSquaresEvalFunc(), DiagonalControlEvalFunc(),
                 CenterControlEvalFunc(), DoublePawnsEvalFunc(), MobilityEvalFunc(), 
                  BothBishopsEvalFunc(), NotAllPawnsEvalFunc(), PinEvalFunc(),
                  AttackerEvalFunc(), HasQueenEvalFunc()]


# ## Collective Evaluation Functions

# In[5]:


class VectorEvaluator(EvalFunc):
    """
        Returns a list/vector where each value is a feature from the EVAL_FUNC_LIST
    """
    def apply(self, board, player):
        scores = []
        for ef in EVAL_FUNC_LIST:
            scores+=[ef.apply(board,player)]
        return scores


class NaiveSumEvaluator(EvalFunc):
    """
        Evaluates board by taking a naive sum of features.x 
    """
    def apply(self, board, player):
        score = 0
        for ef in EVAL_FUNC_LIST:
            score+=ef.apply(board,player)
            
class IntelligentEvaluator(EvalFunc):
    """
        Evaluates Board based on a machine learning model.
    """
    def __init__(self, model):  
        self.model = model
    def apply(self, board, player):
        pass
        
    


# ## Utility Functions               

# In[6]:


def get_samples(pgn_string):   
    """
        This function will take in a pgn_string, extract all the moves played, and then randomly
        sample 10% of the game states and return these states as fen strings.
    """
    pgn = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn)
    
    moves = list(game.mainline_moves())
    
    num_moves = len(moves)
    
    num_samples = num_moves // 5
    
    samples = []
    sample_indices = random.sample(range(num_moves), k=num_samples)
    board = chess.Board()
    for i in range(len(moves)):
        move = moves[i]
        board.push(move)
        if i in sample_indices:
            samples.append(board.fen())
    
    return samples, sample_indices, num_moves  

def get_winner(pgn):
    """
        Returns winner of this game or "Draw" if the game ended in a draw.
    """
    
    try:
        start = pgn.index("Result") + 8
        end = pgn.index("]", start) -1
        result = pgn[start:end]
        if result == "1-0":
            return "White"
        if result == "0-1":
            return "Black"
        if result == "1/2-1/2":
            return "Draw"
    
    except:
        return None

def get_column_names():
    eval_funcs_white = [("W_" + ef.name) for ef in EVAL_FUNC_LIST]
    eval_funcs_black = [("B_" + ef.name) for ef in EVAL_FUNC_LIST]
    eval_funcs_diff = [("DIFF_" + ef.name) for ef in EVAL_FUNC_LIST]

    cols = ["fen", "winner", "move number", "total moves"]
    cols.extend(eval_funcs_white)
    cols.extend(eval_funcs_diff)
    cols.extend(eval_funcs_black)
    return cols
    


# In[7]:


#iterating over games dataset, to create state datast
def populate(games, num_workers):
    """
        For each game, this function will sample 10% of the states and evaluate the features and return
        a DataFrame where each row corresponds to a state and its associated features. It will use
        multiprocessing based on the num_workers argument which will specify how many processes to use.
    
    """
    
    def split(df, n):
        """
            Splits this dataframe into a list of n dataframes for parallel processing
        """
        result = []
        increment = len(df)//n
        current_index = increment
        result.append(df[:current_index])
        for i in range(n-2):
            df_new = df[current_index: current_index + increment]
            current_index+=increment
            result.append(df_new)
        result.append(df[current_index:])
        return result

    
    
    def populate_helper(procnum, df_in, return_dict):
        """
            Iterates over all the games in df_in, samples 10% of the states, and then evaluates all the features.
            It then puts the result datframe in return_dict.
        """
        evaluator = VectorEvaluator()
        result = pd.DataFrame(columns=get_column_names())
        for i in tqdm(range(len(df_in))):

            pgn_string = df_in.iloc[i][1]
            try:
                winner = get_winner(pgn_string)
                samples, sample_indices, num_moves = get_samples(pgn_string)
            
                for i in range(len(samples)):  
                    if samples[i] not in result["fen"]:
                        column_values = []
                        column_values.append(samples[i])
                        column_values.append(winner)
                        column_values.append(sample_indices[i])
                        column_values.append(num_moves)
                        board = chess.Board(fen=samples[i])
                        eval_scores_white = evaluator.apply(board, WHITE)
                        eval_scores_black = evaluator.apply(board, BLACK)
                        eval_scores_diff = [eval_scores_white[i] - eval_scores_black[i] for i in range(len(eval_scores_black))]
                       # eval_scores_black = [-score for score in eval_scores_black]

                        column_values.extend(eval_scores_white)
                        column_values.extend(eval_scores_diff)

                        column_values.extend(eval_scores_black)
                        result.loc[len(result.index)] = column_values
#                         diff = [eval_scores_white[i] - eval_scores_black[i] for i in range(len(eval_scores_white))]
#                         column_values.extend(eval_scores_white)
#                         column_values.extend(eval_scores_black)
#                         result.loc[len(result.index), "White"] = eval_scores_white
#                         result.loc[len(result.index), "Black"] = eval_scores_black
#                         result.loc[len(result.index), "Differene"] = diff

        
                    else:
                        print("dup found")
            except Exception as inst:
                print(inst)
                continue
#                     print(inst)  
#                     print("Error", i)
                    
        return_dict[procnum] = result
        return
    
    

    games_list = split(games, num_workers)
    jobs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    #these are the processes being started
    for i in range(num_workers):
        p = multiprocessing.Process(target=populate_helper, args=(i, games_list[i], return_dict))
        jobs.append(p)
        p.start()
    #waiting for the processes to finish
    for proc in jobs:
        proc.join()

    #building the final complete dataframe from the parallel dataframes
    final_df = pd.DataFrame(columns=get_column_names())
    for i in range(num_workers):
        final_df = final_df.append(return_dict[i])

    return final_df.reset_index(drop=True) 
    


# In[8]:


start_time = time.time()
df_pgn = pd.read_csv("./chess_games_2.csv")

df_pgn = df_pgn.sample(100)
#assert len(df_pgn) == 100000

df_pgn["WINNER"] = df_pgn["pgn"].apply(get_winner)
df_pgn = df_pgn.dropna()

X = df_pgn["pgn"]
y = df_pgn["WINNER"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True)


print("Number of games: ", len(X_train) + len(X_test))

df_eval_X_train = populate(X_train, NUM_PROCESSES)
df_eval_X_train.to_csv("X_train.csv", index = False)
print("\n\n************* X_train completed and saved ***************", len(df_eval_X_train))

df_eval_X_test = populate(X_test, NUM_PROCESSES)
df_eval_X_test.to_csv("X_test.csv", index=False)
print("\n\n************ X_test completed and saved ****************", len(df_eval_X_test))


#df_eval_y_train = populate(y_train, NUM_PROCESSES
y_train.to_csv("y_train.csv", index=False)
print("\n\n*********** y_train completed and saved ****************", len(y_train))


#df_eval_y_test = populate(y_test, NUM_PROCESSES)
y_test.to_csv("y_test.csv", index=False)
print("*********DONE**********", len(y_test))


time = time.time() - start_time
print(time, "seconds")
print(time // 60, "minutes")
print(time // (60**2), "hours")




# In[9]:


# df_eval[[("DIFF_" + ef.name) for ef in EVAL_FUNC_LIST]]


# In[12]:



# In[13]:


len(csv)


# In[14]:



# In[ ]:




