# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from argparse import ArgumentParser
import json 
import numpy as np
import os
import time
import torch

from src.agents import init_agent
from src.environment import Environment
from src.get_data import load_data
from src.run import Run
from src.utilities import create_directory_tree, init_scaler, init_portfolio

def main(args):

    # 0. Preparing the initialization of the training or testing process
    # ########################################################################################
    # 0.1 Specifying the hardware
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 0.2 Init the random seeds for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 0.3 Init all the necessary directory tree structure for efficient logging
    checkpoint_directory = create_directory_tree(mode=args.mode,
                                                 experimental=args.experimental,
                                                 checkpoint_directory=args.checkpoint_directory)
    
    # 0.4 Save the (hyper)parameters used for future reference
    params_dict = vars(args)
    with open(os.path.join(checkpoint_directory, args.mode+"_parameters.json"), "w") as f: 
        json.dump(params_dict, f, indent=4)
    

    # 1. Prepare data
    # ########################################################################################
    # 1.1 Download, process and load the data (stocks from yfinance)
    # df is dataframe (index=dates, columns=stocks, values=closes)
    df = load_data(initial_date=args.initial_date, 
                   final_date=args.final_date, 
                   tickers_subset=args.assets_to_trade,
                   mode=args.mode,
                   path=args.assets_to_trade
                   )
    
    if args.mode == 'test' and args.initial_date is not None and args.final_date is not None:
        df = df.loc[args.initial_date:args.final_date]
        dates = df.index
    

    # 2. Init the trading environment, the data standard scaler and the trading agent
    # ########################################################################################
    
    # 2.1 Preparing the initial portfolio to pass it to the constructor of the environment 
    if args.initial_cash: 
        initial_portfolio = args.initial_cash
    else:
        initial_portfolio = args.initial_portfolio
    tickers = df.columns.to_list()
    # Initial portfolio is a dictionary with keys = tickers and values = number of shares owned
    # - And Bank_account is the key for the cash in bank
    initial_portfolio = init_portfolio(initial_portfolio, tickers)
     
    # 2.2 Init the trading environment   
    # - The environment is a wrapper around the data
    #   - State-space: 
    #       - cash, 
    #       - n_stocks_shares_owned, 
    #       - n_stocks_close_prices 
    #       - optional(n_stocks_corr_matrix, n_stocks_eigenvalues))
    #   - Action-space:
    #      - n_stocks_shares_to_buy
    #      - n_stocks_shares_to_sell
    #   - Reward function:
    #       - The reward is the return of the portfolio
    #       - The return is the sum of the returns of the assets weighted by the number of shares owned
    #       - The return of an asset is the difference between 
    #           - the close price of the asset at the current time step 
    #           - and the close price of the asset at the previous time step

    env = Environment(stock_market_history=df,
                      initial_portfolio=initial_portfolio,
                      buy_cost=args.buy_cost,
                      sell_cost=args.sell_cost,
                      bank_rate=args.bank_rate,
                      limit_n_stocks=args.limit_n_stocks,
                      buy_rule=args.buy_rule,
                      use_corr_matrix=args.use_corr_matrix,
                      use_corr_eigenvalues=args.use_corr_eigenvalues,
                      window=args.window,
                      number_of_eigenvalues=args.number_of_eigenvalues)
    
    # 2.3 Init the data standard scaler
    scaler = init_scaler(env=env, 
                                mode=args.mode,
                                checkpoint_directory=checkpoint_directory)
    
    # 2.4 Init the trading agent
    agent = init_agent(env=env, 
                              device=device, 
                              checkpoint_directory=checkpoint_directory,
                              args=args)
    

    # 3. Run the training or testing process
    # ########################################################################################
    
    # 3.1 Running the whole training or testing process   
    run = Run(env=env,
              agent=agent,
              n_episodes=args.n_episodes,
              agent_type=args.agent_type,
              checkpoint_directory=checkpoint_directory,
              mode=args.mode,
              sac_temperature=args.sac_temperature,
              scaler=scaler)
    
    # 3.2 Saving plots
    initial_time = time.time()
    run.run()
    if args.plot:
        run.logger.generate_plots()
    final_time = time.time()
    print('\nTotal {} duration: {:*^13.3f}\n'.format(args.mode, final_time-initial_time))


if __name__ == '__main__':
    
    parser = ArgumentParser()

    # I. Trading environment parameters
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--initial_cash',      type=float, default=None,         help='Initial cash in the bank, assuming no shares are owned')
    group1.add_argument('--initial_portfolio', type=str,   default='./portfolios_and_tickers/initial_portfolio.json', help='Path to json file containing the content of an initial portfolio, including the cash in bank')
    parser.add_argument('--assets_to_trade',   type=str,   default='./portfolios_and_tickers/tickers_S&P500.txt',     help='List of the tickers of the assets to be traded')
    parser.add_argument('--buy_rule',          type=str,   default='most_first', help="In which order to buy the share: 'most_first' or 'cyclic' or 'random'")
    parser.add_argument('--buy_cost',          type=float, default=0.001,        help='Cost for buying a share, prorata of the quantity being bought')
    parser.add_argument('--sell_cost',         type=float, default=0.001,        help='Cost for selling a share, prorata of the quantity being sold')
    parser.add_argument('--bank_rate',         type=float, default=0.5,          help='Annual bank rate')
    parser.add_argument('--initial_date',      type=str,   default='2014-12-31', help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--final_date',        type=str,   default='2020-12-30', help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")
    parser.add_argument('--limit_n_stocks',    type=int,   default=20,           help='Maximal number of shares that can be bought or sell in one trade')
    
    # 2. Agent parameters
    parser.add_argument('--agent_type',      type=str,   default='distributional', help="Type of agent: 'manual_temperature' or 'automatic_temperature' or 'distributional'")
    parser.add_argument('--sac_temperature', type=float, default=2.0,              help="Coefficient of the entropy term in the loss function in case 'manual_temperature' agent is used")
    
    # 3. Training process parameters
    parser.add_argument('--gamma',       type=float, default=0.99,    help='Discount factor in the definition of the return')
    parser.add_argument('--lr_Q',        type=float, default=0.0003,  help='Learning rate for the critic networks')
    parser.add_argument('--lr_pi',       type=float, default=0.0003,  help='Learning rate for the actor networks')
    parser.add_argument('--lr_alpha',    type=float, default=0.0003,  help='Learning rate for the automatic temperature optimization')
    parser.add_argument('--tau',         type=float, default=0.005,   help='Hyperparameter for the smooth copy of the various networks to their target version')
    parser.add_argument('--batch_size',  type=int,   default=32,      help='Batch size when sampling from the replay buffer')
    parser.add_argument('--memory_size', type=int,   default=100, help='Size of the replay buffer, memory of the agent')
    parser.add_argument('--grad_clip',   type=float, default=1.0,     help='Bound in case one decides to use gradient clipping in the training process')
    parser.add_argument('--delay',       type=int,   default=1,       help='Delay between training of critic and actor')
    
    # 3.2. Model-architecture hyper-parameters
    parser.add_argument('--layer_size', type=int, default=256, help='Number of neurons in the various hidden layers')
    
    # 3.3. Number of training or testing episodes and mode
    parser.add_argument('--n_episodes',   type=int, required=True, help='Number of training or testing episodes')
    parser.add_argument('--mode',         type=str, required=True, help="Mode used: 'train' or 'test'")
    parser.add_argument('--experimental', action='store_true',     help='Saves all outputs in an overwritten directory, used simple experiments and tuning')
    
    # 3.4. random seed, logs information and hardware
    parser.add_argument('--checkpoint_directory', type=str,            default=None,         help='In test mode, specify the directory in which to find the weights of the trained networks')
    parser.add_argument('--plot',                 action='store_true', default=False,        help='Whether to automatically generate plots or not')
    parser.add_argument('--seed',                 type=int,            default='42',         help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices',          type=int, nargs='+', default=[0, 1, 2, 3], help='Specify the GPUs if any')
    
    # II. Data preprocessing parameters
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--use_corr_matrix',       action='store_true', default=False, help='To append the sliding correlation matrix to the time series')
    group2.add_argument('--use_corr_eigenvalues',  action='store_true', default=False, help='To append the eigenvalues of the correlation matrix to the time series')
    parser.add_argument('--window',                type=int,            default=20,    help='Window for correlation matrix computation')
    parser.add_argument('--number_of_eigenvalues', type=int,            default=10,    help='Number of largest eigenvalues to append to the close prices time series')
     
    args = parser.parse_args()
    main(args)
