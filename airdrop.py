from web3 import Web3
import os
import re
from collections import defaultdict
import pandas as pd
from typing import Dict
import numpy as np
from lp_balances import PearlV2Pool, AlmPositions
import datetime as dt
import time
import pickle
from hexbytes import HexBytes
from address_and_contracts import contracts, abis, ROUTE_TO_USTB, address_to_symbol, BLACKLIST, REETH_ADDRESS
import requests
from random import randint
import pickle

Q96 = 2**96
SWP_PCNT_CAP = 0.05
SWP_PTS_MULTIPLIER = 0.75

REAL_RPC_ENDPOINT = 'https://real.drpc.org'
W3 = Web3(Web3.HTTPProvider(REAL_RPC_ENDPOINT, request_kwargs={'timeout': 10}))

NULL_ADDR = '0x0000000000000000000000000000000000000000'
DEFAULT_MULTILIER = 2
LEVERAGE_MULTIPLIER = {
    contracts['ukre'].address: 5,
    contracts['arcusd'].address: 4,
    contracts['pta'].address: 4
}

REBASE_TOKENS = ['ustb', 'arcusd', 'ukre', 'dai']
VE_TOKENS = [('verwa', 'rwa'), ('vepearl', 'pearl')]

TupleToPearlPool = {}
Block_Time_Cache = {}

TOKEN_BY_SEASON = {
    0: [66_600, 46, dt.date(2024, 5, 16), dt.date(2024, 6, 30)],
    1: [1_265_400, 62, dt.date(2024, 7, 1), dt.date(2024, 8, 31)],
    # 2: [999_000, 61, dt.date(2024, 9, 1), dt.date(2024, 10, 31)],
    # 3: [666_000, 61, dt.date(2024, 11, 1), dt.date(2024, 12, 31)],
    # 4: [333_000, 59, dt.date(2025, 1, 1), dt.date(2025, 2, 28)],
}

###############################################################################
HEX_NULL = HexBytes('0x')
def isContract(address):
    while True:
        try:
            return W3.eth.get_code(address) != HEX_NULL
        except:
            print(f'failed to get code for {address}, retrying...')
            time.sleep(1)
            continue

def day_to_season(dt_date):
    if dt_date < TOKEN_BY_SEASON[0][2]:
        return 0
    for season, (_, _, start_date, end_date) in TOKEN_BY_SEASON.items():
        if start_date <= dt_date <= end_date:
            return season
    raise Exception(f'date {dt_date} not in any season')


def get_block_time(block_num, w3_chain=W3):
    global W3
    if block_num not in Block_Time_Cache:
        while True:
            for _ in range(10):
                try:
                    timestamp = w3_chain.eth.get_block(block_num)['timestamp']
                    Block_Time_Cache[block_num] = timestamp
                    return timestamp
                except Exception as e:
                    print(e)
                    print(f'failed to get block time for block {block_num}, retrying...')
                time.sleep(1)
            W3 = Web3(Web3.HTTPProvider(REAL_RPC_ENDPOINT))
    else:
        timestamp = Block_Time_Cache[block_num]
        return timestamp

def _get_block_num_from_ts(target_ts, w3_chain, from_blk=1, to_blk='latest'):
    if to_blk == 'latest':
        to_blk = w3_chain.eth.block_number
    from_time = get_block_time(from_blk, w3_chain)
    # assert(from_time <= target_ts), 'target timestamp earlier than from blk timestamp'
    if from_time > target_ts:
        return from_blk, from_time, from_blk, from_time, from_time
    elif from_time == target_ts:
        return from_blk, from_time, from_blk, from_time, target_ts
    to_time = get_block_time(to_blk, w3_chain)
    assert(to_time >= target_ts), 'target timestamp later than to_blk timestamp'
    if to_time == target_ts:
        return to_blk, to_time, to_blk, to_time, to_time
    while from_blk + 1 < to_blk:
        mid_blk = (from_blk + to_blk) >> 1
        mid_time = get_block_time(mid_blk, w3_chain)
        if mid_time == target_ts:
            return mid_blk, mid_time, mid_blk, mid_time, target_ts
        elif mid_time < target_ts:
            from_blk = mid_blk
        else:
            to_blk = mid_blk
    return from_blk, get_block_time(from_blk, w3_chain), to_blk, get_block_time(to_blk, w3_chain), target_ts

def test_get_block_num_from_ts():
    import time
    from_blk, from_blk_ts, to_blk, to_blk_ts, target_ts = _get_block_num_from_ts(target_ts=time.time()-3600, w3_chain=W3)
    print(f'from_blk_num: {from_blk}, from_blk_ts: {from_blk_ts}, to_blk_num: {to_blk}, to_blk_ts: {to_blk_ts}, target_ts: {target_ts}')

def get_events(contract_event, from_blk, to_blk, arg_filter=None, extra_args=None, blacklist_args=None) -> Dict:    
    if blacklist_args is None:
        blacklist_args = dict()
    if arg_filter is None:
        arg_filter = {}
    res = defaultdict(list)
    while from_blk < to_blk:
        to_blk_limit = to_blk
        while True:
            try:
                events = contract_event.create_filter(
                    fromBlock=from_blk,
                    toBlock=to_blk_limit,
                    argument_filters=arg_filter
                ).get_all_entries()
                break
            except ValueError as e:
                m = re.search(r'range should work: \[(0x[0-9a-f]*), (0x[0-9a-f]*)\]', str(e))
                to_blk_limit = int(m.group(2), 0) if m is not None else (to_blk_limit - from_blk) // 2 + from_blk
            except Exception as e:
                _tmp_events = contract_event.create_filter(
                    fromBlock=from_blk,
                    toBlock=to_blk_limit
                ).get_all_entries()
                print(e)
                print('events:', _tmp_events)
                to_blk_limit = from_blk + (to_blk_limit - from_blk) >> 1
                # raise Exception(f'failed to get events from {from_blk} to {to_blk}')
        for event in events:
            res['block'].append(event['blockNumber'])
            for key, val in event['args'].items():
                if key in blacklist_args:
                    if blacklist_args[key] is None or val in blacklist_args[key]:
                        continue
                res[key].append(val)
            if extra_args is not None:
                for extra_arg in extra_args:
                    res[extra_arg].append(event[extra_arg])
        from_blk = to_blk_limit + 1
    return res

def _get_new_pearl_pools(from_blk=1, to_blk=None, w3_chain=W3, pools_by_address=None, print_symbol=False):
    if to_blk is None:
        to_blk = w3_chain.eth.block_number
    if pools_by_address is None:
        pools_by_address = {}
    print(f'get {len(pools_by_address)} pools before block {from_blk} by input pools_by_address')
    for pool_address, pearl_pool in pools_by_address.items():
        _t0, _t1, _fee = pearl_pool.token0, pearl_pool.token1, pearl_pool.fee
        PearlV2Pool.add_pool((_t0, _t1, _fee), pool_address)
        # print(f'add pool address {pool_address} with tokens {_t0}, {_t1}, fee {_fee} at block {from_blk}')
        pearl_pool.latest_sqrtPriceX96, pearl_pool.latest_tick, _, _, _, _, _ = pearl_pool.contract.functions.slot0().call(block_identifier=max(from_blk-1, pearl_pool.init_blk))
    crt_events = get_events(contracts['pearl_factory'].events.PoolCreated, from_blk, to_blk, blacklist_args={'tickSpacing':None})
    for i in range(len(crt_events['block'])):
        _t0, _t1, _fee, _pool_address, _init_blk = crt_events['token0'][i], crt_events['token1'][i], crt_events['fee'][i], crt_events['pool'][i], crt_events['block'][i]
        _pool_contract = w3_chain.eth.contract(_pool_address, abi=abis['pearlv2_pool'])
        sqrtPriceX96, tick, _, _, _, _, _ = _pool_contract.functions.slot0().call(block_identifier=_init_blk)
        print(f'find new pool address {_pool_address} with tokens {_t0}, {_t1}, fee {_fee} at block {_init_blk}')
        if crt_events['token0'][i] not in ROUTE_TO_USTB:
            print(_t0, _t1, _fee, _pool_address, _init_blk)
            raise Exception(f'token {_t0} not in route to USTB')
        if crt_events['token1'][i] not in ROUTE_TO_USTB:
            print(_t0, _t1, _fee, _pool_address, _init_blk)
            raise Exception(f'token {_t1} not in route to USTB')
        PearlV2Pool.add_pool((_t0, _t1, _fee), _pool_address)
        pools_by_address[_pool_address] = PearlV2Pool(
            _t0, _t1, _fee, sqrtPriceX96, tick, _pool_contract, _init_blk
        )
        TupleToPearlPool[(_t0, _t1, _fee)] = pools_by_address[_pool_address]
    if print_symbol:
        for pool_address, pearl_pool in pools_by_address.items():
            t0_contract = W3.eth.contract(pearl_pool.token0, abi=abis['erc20'])
            t0_symbol = t0_contract.functions.symbol().call()
            t1_contract = W3.eth.contract(pearl_pool.token1, abi=abis['erc20'])
            t1_symbol = t1_contract.functions.symbol().call()
            print(f'pool address:{pool_address}, latest block: {pearl_pool.init_blk}')
            print(f't0:{t0_symbol}({pearl_pool.token0}), t1:{t1_symbol}({pearl_pool.token1}), fee:({pearl_pool.fee})')            
    return pools_by_address

def getPearlPoolSwap(from_blk, to_blk, pearl_pools=None, w3_chain=W3):
    if pearl_pools is None:
        pearl_pools = _get_new_pearl_pools(from_blk=from_blk, to_blk=to_blk, w3_chain=w3_chain)
    df_swap_by_address = {}
    df_pool_price_by_address = {}
    for _pool_address, pearl_pool in pearl_pools.items():
        swap_events = get_events(
            pearl_pool.contract.events.Swap, from_blk, to_blk,
            extra_args=['transactionIndex', 'logIndex'],
            blacklist_args={'liquidity':None}
        )
        print(f'daily starting price for {_pool_address} {pearl_pool.latest_sqrtPriceX96}')
        swap_events['block'] = [from_blk-1] + swap_events['block']
        swap_events['sqrtPriceX96'] = [pearl_pool.latest_sqrtPriceX96] + swap_events['sqrtPriceX96']
        swap_events['tick'] = [pearl_pool.latest_tick] + swap_events['tick']
        swap_events['transactionIndex'] = [0] + swap_events['transactionIndex']
        swap_events['logIndex'] = [0] + swap_events['logIndex']
        swap_events['sender'] = [''] + swap_events['sender']
        swap_events['recipient'] = [''] + swap_events['recipient']
        swap_events['amount0'] = [0] + swap_events['amount0']
        swap_events['amount1'] = [0] + swap_events['amount1']

        df_swap = pd.DataFrame(swap_events)
        df_swap['block'] = df_swap['block'].astype('int64')
        df_swap['transactionIndex'] = df_swap['transactionIndex'].astype('int64')
        df_swap['logIndex'] = df_swap['logIndex'].astype('int64')
        df_swap = df_swap.sort_values(['block', 'transactionIndex', 'logIndex'])
        df_swap['sqrtPriceX96'] = df_swap['sqrtPriceX96'].astype('float')
        df_swap['amount0'] = df_swap['amount0'].astype('float')
        df_swap['amount1'] = df_swap['amount1'].astype('float')
        df_swap_by_address[_pool_address] = df_swap.copy()
        df_swap['ix'] = df_swap.apply(lambda row: (row.block<<20) + (row.transactionIndex<<10) + row.logIndex, axis=1)
        df_swap.sort_values('ix', inplace=True)
        df_pool_price = df_swap.groupby('block').last()[['sqrtPriceX96', 'tick']]
        df_pool_price['timestamp'] = df_pool_price.index.map(lambda x: get_block_time(x, w3_chain))
        df_pool_price.query('sqrtPriceX96 > 0', inplace=True)
        df_pool_price_by_address[_pool_address] = df_pool_price

        pearl_pool.latest_sqrtPriceX96 = df_swap['sqrtPriceX96'].iloc[-1]
        pearl_pool.latest_tick = df_swap['tick'].iloc[-1]
    return df_swap_by_address, df_pool_price_by_address

def update_df_price_by_token(token_contract, from_blk, to_blk, df_pool_price_by_address, w3_chain=W3):
    df_price = pd.DataFrame(index=[from_blk-1, to_blk])
    # df_price['p'] = np.nan
    for path_addr in ROUTE_TO_USTB[token_contract.address]:
        pool_addr = path_addr if path_addr[0] == '0' else '0' + path_addr[1:]
        if pool_addr in df_pool_price_by_address:
            df_swap = df_pool_price_by_address[pool_addr]
            if len(df_swap) == 0 or df_swap.index[-1] < from_blk or df_swap.index[-1] > to_blk:
                continue
            val = (df_swap.sqrtPriceX96 / Q96)**2 if path_addr[0] == '0' else (Q96 / df_swap.sqrtPriceX96)**2
            df_price = df_price.merge(val, how='outer', left_index=True, right_index=True)
    df_price = df_price.astype(float).ffill().product(axis=1).rename('px')
    df_price = df_price.to_frame()
    token_name = address_to_symbol[token_contract.address]
    if token_name in REBASE_TOKENS:
        try:
            last_rebase_index = contracts[token_name].functions.rebaseIndex().call(block_identifier=from_blk-1)
        except:
            last_rebase_index = 1e18
            print(f'rebase index no ready for {token_name} @ {from_blk-1}, set to 1')
        rebase_events = get_events(
            token_contract.events.RebaseIndexUpdated, from_blk, to_blk,
            extra_args=['transactionIndex', 'logIndex'],
            blacklist_args={'updatedBy':None, 'totalSupplyBefore':None, 'totalSupplyAfter':None}
        )
        rebase_events['block'] = [from_blk-1] + rebase_events['block']
        rebase_events['index'] = [last_rebase_index] + rebase_events['index']
        rebase_events['transactionIndex'] = [0] + rebase_events['transactionIndex']
        rebase_events['logIndex'] = [0] + rebase_events['logIndex']
        df_rebase = pd.DataFrame(rebase_events)
        df_rebase['index'] = df_rebase['index'].astype('float')/1e18
        df_price = df_price.merge(df_rebase.set_index('block')['index'], how='outer', left_index=True, right_index=True)
        df_rebase['ix'] = df_rebase.apply(lambda row: (int(row.block)<<20) + (int(row.transactionIndex)<<10) + int(row.logIndex), axis=1)
    else:
        df_price['index'] = 1.0
        df_rebase = None
    df_price = df_price.ffill()
    df_price['timestamp'] = df_price.index.map(lambda x: get_block_time(x, w3_chain)).values
    return df_price, df_rebase


def getAllRebaseEvents(
        from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
        df_token_price_by_blk, df_rebase_by_address, last_shares_by_address
    ):
    total_time = to_ts - from_ts
    weighted_avg = {}
    for token_name in REBASE_TOKENS:
        if contracts[token_name].address not in last_shares_by_address:
            last_shares_by_address[contracts[token_name].address] = {}
    
        weighted_avg[token_name] = defaultdict(int)
        token_contract = contracts[token_name]
        df_price = df_token_price_by_blk[token_contract.address].copy()
        df_price.px *= df_price['index']
        if len(last_shares_by_address[token_contract.address]) > 0:
            for addr, (shs, last_idx, last_blk) in last_shares_by_address[token_contract.address].items():
                if blk_after_from - 1 != df_price.index[0]:
                    print('first row of price df should be blk_after_from-1')
                    print(df_price.head(5))
                    print(blk_after_from)
                    raise
                _wtd = df_price.iloc[0].px * (blk_ts_after_from - from_ts) / total_time * shs
                if addr != NULL_ADDR and _wtd < 0:
                    print('negative weighted average')
                    print(f'{shs}, {from_ts}, {blk_ts_after_from}, {df_price.iloc[0].px}')
                    raise
                weighted_avg[token_name][addr] += _wtd
                last_shares_by_address[token_contract.address][addr][-1] = blk_after_from
        df_rebase = df_rebase_by_address[token_contract.address]
        xfer_events = get_events(token_contract.events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
        # from/to/value
        if len(xfer_events['block']) > 0:
            df_xfers = pd.DataFrame(xfer_events)
            df_xfers['value'] = df_xfers['value'].astype('float')
            df_xfers['ix'] = df_xfers.apply(lambda row: (row.block<<20) + (row.transactionIndex<<10) + (row.logIndex), axis=1)
            df_rebase = pd.merge_asof(df_xfers, df_rebase[['ix', 'index']], on='ix')
        df_rebase['from'] = df_rebase['from'].fillna('') if 'from' in df_rebase else ''
        df_rebase['to'] = df_rebase['to'].fillna('') if 'to' in df_rebase else ''
        df_rebase['value'] = df_rebase['value'].fillna(0) if 'value' in df_rebase else 0
        for _, row in df_rebase.iterrows():
            if row['to'] in last_shares_by_address[token_contract.address]:
                shs, last_idx, last_blk = last_shares_by_address[token_contract.address][row['to']]
                wtd_px = _get_wtd_price_btw_blk(last_blk, row['block'], df_price)
                if wtd_px is not None:
                    weighted_avg[token_name][row['to']] += wtd_px / total_time * shs
                    last_shares_by_address[token_contract.address][row['to']][0] += (row['value'] / row['index'])
                    last_shares_by_address[token_contract.address][row['to']][1] = row['index']
                    last_shares_by_address[token_contract.address][row['to']][2] = row.block
            else:
                if row['to'] not in BLACKLIST:
                    last_shares_by_address[token_contract.address][row['to']] = [row['value']/row['index'], row['index'], row.block]
            if row['from'] in last_shares_by_address[token_contract.address]:
                shs, last_idx, last_blk = last_shares_by_address[token_contract.address][row['from']]
                wtd_px = _get_wtd_price_btw_blk(last_blk, row['block'], df_price)
                if wtd_px is not None:
                    weighted_avg[token_name][row['from']] += wtd_px / total_time * shs
                    last_shares_by_address[token_contract.address][row['from']][0] -= (row['value'] / row['index'])
                    balance_after = last_shares_by_address[token_contract.address][row['from']][0]
                    if row['from'] not in BLACKLIST and balance_after < 0:
                        last_shares_by_address[token_contract.address][row['from']][0] = 0
                    last_shares_by_address[token_contract.address][row['from']][1] = row['index']
                    last_shares_by_address[token_contract.address][row['from']][2] = row.block
            else:
                if row['from'] not in BLACKLIST:
                    print(f'token sent from empty account {row["from"]}')
                    print(row.to_dict())
                    last_shares_by_address[token_contract.address][row['from']] = [0, row['index'], row.block]
        for addr, (shs, last_idx, last_blk) in last_shares_by_address[token_contract.address].items():
            _wtd = _get_wtd_price_btw_blk(last_blk, blk_before_to, df_price)
            if _wtd is not None:
                weighted_avg[token_name][addr] += _wtd / total_time * shs
            _wtd = df_price.iloc[-1].px * (to_ts - blk_ts_before_to) / total_time * shs
            if addr != NULL_ADDR and _wtd < 0:
                raise
            weighted_avg[token_name][addr] += _wtd
            last_shares_by_address[token_contract.address][addr][2] = blk_before_to
    return weighted_avg, last_shares_by_address

def _get_wtd_price_btw_blk(from_blk, to_blk, df_price, w3_chain=W3, debug=False):
    if from_blk >= to_blk:
        return 0
    from_ix = np.searchsorted(df_price.index, from_blk, side='right')
    if from_ix < len(df_price):
        to_ix = np.searchsorted(df_price.index, to_blk, side='left') - 1
        if to_ix >= from_ix:
            wtd_px = df_price.iloc[from_ix-1].px * (df_price.iloc[from_ix].timestamp - get_block_time(from_blk, w3_chain))
            wtd_px += sum(df_price.timestamp.diff().shift(-1).iloc[from_ix:to_ix] * df_price.iloc[from_ix:to_ix].px)
            wtd_px += df_price.iloc[to_ix].px * (get_block_time(to_blk, w3_chain) - df_price.iloc[to_ix].timestamp)
        else:
            wtd_px = df_price.iloc[from_ix-1].px * (get_block_time(to_blk, w3_chain) - get_block_time(from_blk, w3_chain))        
        return wtd_px
    else:
        return None

def _get_ix(row):
    return (int(row.block) << 20) + (int(row.transactionIndex) << 10) + int(row.logIndex)    

def averageTokenInALM(
        from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
        df_token_price_by_blk, df_swap_by_address, running_positions=None, w3_chain=W3
    ):
    twab_by_token = dict() # {token_address: {owner: twab}}
    twab_by_pool = dict() # {pool_address: {t0:twab, t1:twab}}
    total_time = to_ts - from_ts

    liquid_box_manager = contracts['liquid_box_manager']
    df_mgr = []
    event_dp = get_events(
        liquid_box_manager.events.Deposit, blk_after_from, blk_before_to,
        extra_args=['transactionIndex', 'logIndex'], blacklist_args={'amount0':None, 'amount1':None}
    )
    if len(event_dp['block']) > 0:
        df_dp = pd.DataFrame(event_dp)
        df_dp["shares"] = df_dp["shares"].astype('float')
        df_dp['from'] = ''
        df_dp['ix'] = df_dp.apply(_get_ix, axis=1)
        df_dp['event'] = 'deposit'
        df_mgr.append(df_dp[['block', 'from', 'to', 'shares', 'ix', 'box', 'event']])
    event_wd = get_events(liquid_box_manager.events.Withdraw, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'amount0':None, 'amount1':None})
    if len(event_wd['block']) > 0:
        df_wd = pd.DataFrame(event_wd)
        df_wd["shares"] = df_wd["shares"].astype('float')
        df_wd['from'] = df_wd['to']
        df_wd['to'] = ''
        df_wd['ix'] = df_wd.apply(_get_ix, axis=1)
        df_wd['event'] = 'withdraw'
        df_mgr.append(df_wd[['block', 'from', 'to', 'shares', 'ix', 'box', 'event']])
    if len(df_mgr) > 0:
        df_mgr = pd.concat(df_mgr)
    
    liquid_box_factory = contracts['liquid_box_factory']
    event_crt = get_events(liquid_box_factory.events.BoxCreated, 1, blk_before_to)
    time_weights_before_from = (blk_ts_after_from - from_ts) / total_time
    time_weights_after_to = (to_ts - blk_ts_before_to) / total_time
    for i in range(len(event_crt['block'])):
        _box_addr = event_crt['box'][i]
        if len(df_mgr) > 0:
            df_mgr_i = df_mgr.loc[df_mgr['box'] == _box_addr]
            if len(df_mgr_i) > 0:
                df_i = [df_mgr_i]
            else:
                df_i = []
        else:
            df_i = []
        _box_contract = w3_chain.eth.contract(address=_box_addr, abi=abis['liquid_box'])
        _t0 = event_crt['token0'][i]
        _t1 = event_crt['token1'][i]
        if _t0 not in twab_by_token:
            twab_by_token[_t0] = defaultdict(int)
        if _t1 not in twab_by_token:
            twab_by_token[_t1] = defaultdict(int)
        df_px0 = df_token_price_by_blk[_t0]
        df_px1 = df_token_price_by_blk[_t1]
        _fee = event_crt['fee'][i]
        _pool_contract = TupleToPearlPool[(_t0, _t1, _fee)].contract
        twab_by_pool[_pool_contract.address] = {_t0:0, _t1:0}

        df_swap = df_swap_by_address[_pool_contract.address]
        if len(df_swap) > 0:
            df_swap['ix'] = df_swap.apply(lambda row: ((row.block+1)<<20)-1, axis=1)
            df_swap['event'] = 'swap'
            df_i.append(df_swap[['block', 'tick', 'ix', 'event']])
        _event_mnt = get_events(_pool_contract.events.Mint, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"owner": _box_addr}, blacklist_args={'amount0':None, 'amount1':None, 'amount':None, 'owner':None, 'sender':None})
        if len(_event_mnt['block']) > 0:
            df_pool_mnt = pd.DataFrame(_event_mnt)
            df_pool_mnt = df_pool_mnt.rename(columns={'actualLiquidity':'liquidity'})
            df_pool_mnt["liquidity"] = df_pool_mnt["liquidity"].astype('float')
            df_pool_mnt['ix'] = df_pool_mnt.apply(_get_ix, axis=1)
            df_pool_mnt['event'] = 'pool_mint'
            df_i.append(df_pool_mnt[['block', 'tickLower', 'tickUpper', 'liquidity', 'ix', 'event']])
        _event_burn = get_events(_pool_contract.events.Burn, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"owner": _box_addr}, blacklist_args={'amount0':None, 'amount1':None, 'owner':None, 'sender':None})
        if len(_event_burn['block']) > 0:
            df_pool_burn = pd.DataFrame(_event_burn)
            df_pool_burn = df_pool_burn.rename(columns={'amount':'liquidity'})
            df_pool_burn["liquidity"] = df_pool_burn["liquidity"].astype('float')
            df_pool_burn = df_pool_burn[df_pool_burn['liquidity'] > 0]
            if len(df_pool_burn) > 0:
                df_pool_burn['liquidity'] = -df_pool_burn['liquidity']
                df_pool_burn['ix'] = df_pool_burn.apply(_get_ix, axis=1)
                df_pool_burn['event'] = 'pool_burn'
                df_i.append(df_pool_burn[['block', 'tickLower', 'tickUpper', 'liquidity', 'ix', 'event']])
        
        _event_xfer_in0 = get_events(contracts[address_to_symbol[_t0]].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"to": _box_addr}, blacklist_args={'from':None, 'to':None})
        if len(_event_xfer_in0['block']) > 0:
            df_xfer_in0 = pd.DataFrame(_event_xfer_in0)
            df_xfer_in0 = df_xfer_in0.rename(columns={'value':'balance0'})
            df_xfer_in0["balance0"] = df_xfer_in0["balance0"].astype('float')
            df_xfer_in0['ix'] = df_xfer_in0.apply(_get_ix, axis=1)
            df_xfer_in0['event'] = 't0_xfer_to_box'
            df_i.append(df_xfer_in0[['block', 'balance0', 'ix', 'event']])
        _event_xfer_out0 = get_events(contracts[address_to_symbol[_t0]].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"from": _box_addr}, blacklist_args={'from':None, 'to':None})
        if len(_event_xfer_out0['block']) > 0:
            df_xfer_out0 = pd.DataFrame(_event_xfer_out0)
            df_xfer_out0 = df_xfer_out0.rename(columns={'value':'balance0'})
            df_xfer_out0["balance0"] = -df_xfer_out0["balance0"].astype('float')
            df_xfer_out0['ix'] = df_xfer_out0.apply(_get_ix, axis=1)
            df_xfer_out0['event'] = 't0_xfer_from_box'
            df_i.append(df_xfer_out0[['block', 'balance0', 'ix', 'event']])
        _event_xfer_in1 = get_events(contracts[address_to_symbol[_t1]].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"to": _box_addr}, blacklist_args={'from':None, 'to':None})
        if len(_event_xfer_in1['block']) > 0:
            df_xfer_in1 = pd.DataFrame(_event_xfer_in1)
            df_xfer_in1 = df_xfer_in1.rename(columns={'value':'balance1'})
            df_xfer_in1["balance1"] = df_xfer_in1["balance1"].astype('float')
            df_xfer_in1['ix'] = df_xfer_in1.apply(_get_ix, axis=1)
            df_xfer_in1['event'] = 't1_xfer_to_box'
            df_i.append(df_xfer_in1[['block', 'balance1', 'ix', 'event']])
            
        _event_xfer_out1 = get_events(contracts[address_to_symbol[_t1]].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], arg_filter={"from": _box_addr}, blacklist_args={'from':None, 'to':None})
        if len(_event_xfer_out1['block']) > 0:
            df_xfer_out1 = pd.DataFrame(_event_xfer_out1)
            df_xfer_out1 = df_xfer_out1.rename(columns={'value':'balance1'})
            df_xfer_out1["balance1"] = -df_xfer_out1["balance1"].astype('float')
            df_xfer_out1['ix'] = df_xfer_out1.apply(_get_ix, axis=1)
            df_xfer_out1['event'] = 't1_xfer_from_box'
            df_i.append(df_xfer_out1[['block', 'balance1', 'ix', 'event']])
        
        # block/tickLower/tickUpper/transactionIndex/logIndex
        _event_xfer = get_events(_box_contract.events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
        if len(_event_xfer['block']) > 0:
            df_xfer = pd.DataFrame(_event_xfer)
            df_xfer = df_xfer[np.logical_and(df_xfer['from'] != NULL_ADDR, df_xfer['to'] != NULL_ADDR)]
            if len(df_xfer) > 0:
                df_xfer.rename(columns={'value': 'shares'}, inplace=True)
                df_xfer['ix'] = df_xfer.apply(_get_ix, axis=1)
                df_xfer['event'] = 'box_xfer'
                df_xfer["shares"] = df_xfer["shares"].astype('float')
                df_i.append(df_xfer[['block', 'from', 'to', 'shares', 'ix', 'event']])
        _event_cf = get_events(_box_contract.events.CollectFees, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'feesToOwner0':None, 'feesToOwner1':None})
        if len(_event_cf['block']) > 0:
            df_cf = pd.DataFrame(_event_cf)
            df_cf['ix'] = df_cf.apply(_get_ix, axis=1)
            df_cf['event'] = 'collect_fee'
            df_cf['feesToVault0'] = df_cf['feesToVault0'].astype('float')
            df_cf['feesToVault1'] = df_cf['feesToVault1'].astype('float')
            df_i.append(df_cf[['block', 'feesToVault0', 'feesToVault1', 'ix', 'event']])
        
        if _box_addr not in running_positions:
            running_positions[_box_addr] = AlmPositions(_box_addr, _box_contract.functions.gauge().call())
        running_pos = running_positions[_box_addr]

        for owner, bal0, bal1, last_blk in running_pos.get_current_reserves(blk_after_from):
            val0 = bal0 * df_px0.iloc[0].px * time_weights_before_from
            val1 = bal1 * df_px1.iloc[0].px * time_weights_before_from
            twab_by_token[_t0][owner] += val0
            twab_by_token[_t1][owner] += val1
            twab_by_pool[_pool_contract.address][_t0] += val0
            twab_by_pool[_pool_contract.address][_t1] += val1

        if len(df_i) > 0:
            df_i = pd.concat(df_i)
            df_i = df_i.sort_values('ix')
            df_i['shares'] = df_i['shares'].fillna(0) if 'shares' in df_i else 0
            df_i['balance0'] = df_i['balance0'].fillna(0) if 'balance0' in df_i else 0
            df_i['balance1'] = df_i['balance1'].fillna(0) if 'balance1' in df_i else 0
            df_i['liquidity'] = df_i['liquidity'].fillna(0) if 'liquidity' in df_i else 0
            df_i['feesToVault0'] = df_i['feesToVault0'].fillna(0) if 'feesToVault0' in df_i else 0
            df_i['feesToVault1'] = df_i['feesToVault1'].fillna(0) if 'feesToVault1' in df_i else 0
            df_i['tick'] = df_i['tick'].ffill()
            
            for _, row in df_i.iterrows():
                if row['shares'] != 0: # box Deposit/Withdraw/Transfer, shares change
                    last_states = running_pos.box_token_transfer(row['from'], row['to'], row['shares'], row['block'])
                elif row['liquidity'] != 0: # liquidity/tick change
                    last_states = running_pos.pool_change(row['tickLower'], row['tickUpper'], row['liquidity'], row['block'])
                elif row['balance0'] != 0: # balance0 transfer
                    last_states = running_pos.box_balance0_change(row['balance0'], row['block'])
                elif row['balance1'] != 0: # balance1 transfer
                    last_states = running_pos.box_balance1_change(row['balance1'], row['block'])
                elif row['feesToVault0'] != 0 or row['feesToVault1'] != 0: # CollectFees
                    last_states = running_pos.set_box_fees_to_vault(row['feesToVault0'], row['feesToVault1'], row['block'])
                else: # tick change, from pool swap
                    last_states = running_pos.get_current_reserves(row['block'], new_tick=row['tick'])
                for owner, bal0, bal1, last_blk in last_states:
                    if bal0 < 0 or bal1 < 0:
                        print(f'negative balance in averageTokenInALM @ blk({last_blk}), owner({owner}), balance0({bal0}), balance1({bal1})')
                        print(f'event info: {row.to_dict()}')
                        raise
                    if owner in BLACKLIST:
                        continue
                    wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_px0)
                    if wtd_px is not None:
                        assert wtd_px >= 0
                        _val = wtd_px / total_time * bal0
                        twab_by_token[_t0][owner] += _val
                        twab_by_pool[_pool_contract.address][_t0] += _val
                    wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_px1)
                    if wtd_px is not None:
                        assert wtd_px >= 0
                        _val = wtd_px / total_time * bal1
                        twab_by_token[_t1][owner] += _val
                        twab_by_pool[_pool_contract.address][_t1] += _val
        for owner, bal0, bal1, last_blk in running_pos.get_current_reserves(blk_before_to):
            wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=blk_before_to, df_price=df_px0)
            if wtd_px is not None:
                assert wtd_px >= 0
                _val = wtd_px / total_time * bal0
                twab_by_token[_t0][owner] += _val
                twab_by_pool[_pool_contract.address][_t0] += _val
            _val_after_to_blk = (df_px0.iloc[-1].px * time_weights_after_to * bal0)
            twab_by_token[_t0][owner] += _val_after_to_blk
            twab_by_pool[_pool_contract.address][_t0] += _val_after_to_blk
            wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=blk_before_to, df_price=df_px1)
            if wtd_px is not None:
                assert wtd_px >= 0
                _val = wtd_px / total_time * bal1
                twab_by_token[_t1][owner] += _val
                twab_by_pool[_pool_contract.address][_t1] += _val
            _val_after_to_blk = (df_px1.iloc[-1].px * time_weights_after_to * bal1)
            twab_by_token[_t1][owner] += _val_after_to_blk
            twab_by_pool[_pool_contract.address][_t1] += _val_after_to_blk
        running_pos.balance0_left = _box_contract.functions.getBalance0().call(block_identifier=blk_before_to)
        running_pos.balance1_left = _box_contract.functions.getBalance1().call(block_identifier=blk_before_to)
    return twab_by_token, twab_by_pool

def get_gauge_contracts(voter_contract, target_tokens=None, from_blk=1, to_blk='latest', gauge_by_pool=None, w3_chain=W3):
    if to_blk == 'latest':
        to_blk = w3_chain.eth.block_number
    if gauge_by_pool is None:
        gauge_by_pool = dict()
    gauge_created_events = get_events(
        voter_contract.events.GaugeCreated, from_blk, to_blk,
        extra_args=['transactionIndex', 'logIndex'],
        blacklist_args={'external_bribe':None, 'creator':None, 'internal_bribe':None}
    )
    for i in range(len(gauge_created_events['block'])):
        _pool_address = gauge_created_events['pool'][i]
        _pool_contract = w3_chain.eth.contract(address=_pool_address, abi=abis['pearlv2_pool'])
        _t0 = _pool_contract.functions.token0().call(block_identifier=to_blk)
        _t1 = _pool_contract.functions.token1().call(block_identifier=to_blk)
        if target_tokens is None or _t0 in target_tokens or _t1 in target_tokens:
            gauge_by_pool[_pool_address] = [gauge_created_events['gauge'][i], gauge_created_events['block'][i]]
    return gauge_by_pool

def averageTokenInLPNFT(
        from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
        df_swap_by_address, df_price_by_token, pool_ranges_by_address=None, tokens_to_track=None, gauge_by_address=None
    ):
    nft_manager = contracts['pearl_nft_manager']
    dfs_price = {}
    twab_by_token = {}
    if tokens_to_track is not None:
        for token_addr in tokens_to_track:
            dfs_price[token_addr] = df_price_by_token[token_addr]
            twab_by_token[token_addr] = defaultdict(int)
    twab_by_pool = {}
    total_time = to_ts - from_ts
    # Transfer: from/to/tokenId
    df_lp = []
    xfer_events = get_events(nft_manager.events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(xfer_events['block']) > 0:
        df_xfer = pd.DataFrame(xfer_events)
        df_xfer['liquidity'] = 0
        df_xfer['event'] = 'xfer'
        df_lp.append(df_xfer)
    # IncreaseLiquidity: tokenId/liquidity/actualLiquidity/amount0/amount1
    inc_liquidity_events = get_events(nft_manager.events.IncreaseLiquidity, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'liquidity':None, 'amount0':None, 'amount1':None})
    if len(inc_liquidity_events['block']) > 0:
        df_inc = pd.DataFrame(inc_liquidity_events)
        df_inc.rename(columns={'actualLiquidity':'liquidity'}, inplace=True)
        df_inc['liquidity'] = df_inc['liquidity'].astype('float')
        df_inc['event'] = 'inc_liq'
        df_lp.append(df_inc)
    # DecreaseLiquidity: tokenId/liquidity/amount0/amount1
    dec_liquidity_events = get_events(nft_manager.events.DecreaseLiquidity, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'amount0':None, 'amount1':None})
    if len(dec_liquidity_events['block']) > 0:
        df_dec = pd.DataFrame(dec_liquidity_events)
        df_dec['liquidity'] = df_dec['liquidity'].astype('float')
        # df_dec['liquidity'] = -df_dec['liquidity']
        df_dec['event'] = 'dec_liq'
        df_lp.append(df_dec)
    if len(df_lp) > 0:
        df_lp = pd.concat(df_lp)
        df_lp['pool'] = df_lp.apply(lambda row: PearlV2Pool.get_pool_by_tokenId(row.tokenId, row.block+1), axis=1)
    
    time_weight_before_from_blk = (blk_ts_after_from - from_ts) / total_time
    time_weight_after_to_blk = (to_ts - blk_ts_before_to) / total_time
    for pool_address, pool_ranges in pool_ranges_by_address.items():
        gauge_address = gauge_by_address[pool_address][0] if (gauge_by_address is not None and pool_address in gauge_by_address) else ""
        twab_by_pool[pool_address] = {pool_ranges.token0:0, pool_ranges.token1:0}
        if tokens_to_track is not None:
            if pool_ranges.token0 not in tokens_to_track and pool_ranges.token1 not in tokens_to_track:
                continue
            if pool_ranges.token0 in tokens_to_track:
                df_price0 = df_price_by_token[pool_ranges.token0]
                twab_by_token[pool_ranges.token0] = defaultdict(int)
            if pool_ranges.token1 in tokens_to_track:
                df_price1 = df_price_by_token[pool_ranges.token1]
                twab_by_token[pool_ranges.token1] = defaultdict(int)
        else:
            df_price0 = df_price_by_token[pool_ranges.token0] if pool_ranges.token0 in df_price_by_token else None
            df_price1 = df_price_by_token[pool_ranges.token1] if pool_ranges.token1 in df_price_by_token else None
            twab_by_token[pool_ranges.token0] = defaultdict(int)
            twab_by_token[pool_ranges.token1] = defaultdict(int)
        for addr, res0, res1, _, gauged in pool_ranges.get_current_reserves(latest_blk_num=blk_after_from): # set last blk to from_blk
            if addr in BLACKLIST:
                continue
            if res0 < 0 or res1 < 0:
                print(f'negative balance in averageTokenInLPNFT @ blk({blk_after_from}), owner({addr}), res0({res0}), res1({res1})')
                raise
            if df_price0 is not None:
                _wtd0 = df_price0.iloc[0].px * time_weight_before_from_blk * res0
                assert _wtd0 >= 0
                twab_by_pool[pool_address][pool_ranges.token0] += _wtd0
                twab_by_token[pool_ranges.token0][addr] += _wtd0
            if df_price1 is not None:
                _wtd1 = df_price1.iloc[0].px * time_weight_before_from_blk * res1
                assert _wtd1 >= 0
                twab_by_pool[pool_address][pool_ranges.token1] += _wtd1
                twab_by_token[pool_ranges.token1][addr] += _wtd1
        
        df_xfer = df_swap_by_address[pool_address].reset_index()
        if len(df_lp) > 0:
            df_lp_i = df_lp[df_lp.pool == pool_address]
            if len(df_lp_i) > 0:
                df_lp_i.loc[:,'ix'] = df_lp_i.apply(_get_ix, axis=1)
                df_xfer['ix'] = df_xfer.apply(lambda row: ((row.block+1)<<20)-1, axis=1)
                df_xfer = pd.concat([df_xfer, df_lp_i])
                df_xfer = df_xfer.sort_values('ix')
        df_xfer['tokenId'] = df_xfer['tokenId'].fillna(-1) if 'tokenId' in df_xfer else -1
        for _, row in df_xfer.iterrows():
            res_tuple = None
            if row['tokenId'] == -1:
                for addr, res0, res1, last_blk, _ in pool_ranges.tick_change(row['tick'], row['block']):
                    if addr in BLACKLIST:
                        continue
                    if res0 < 0 or res1 < 0:
                        print(f'negative balance in averageTokenInLPNFT @ blk({last_blk}), owner({addr}), res0({res0}), res1({res1})')
                        raise
                    if df_price0 is not None:
                        wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_price0)
                        if wtd_px is not None:
                            assert wtd_px >= 0
                            _val = wtd_px / total_time * res0
                            twab_by_pool[pool_address][pool_ranges.token0] += _val
                            twab_by_token[pool_ranges.token0][addr] += _val
                    if df_price1 is not None:
                        wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_price1)
                        if wtd_px is not None:
                            assert wtd_px >= 0
                            _val = wtd_px / total_time * res1
                            twab_by_token[pool_ranges.token1][addr] += _val
                            twab_by_pool[pool_address][pool_ranges.token1] += _val
            else:
                if row['event'] == 'xfer': # Transfer
                    res_tuple = pool_ranges.transfer_token(row['tokenId'], row['to'], row['to'] == gauge_address) # None if new mint
                elif row['event'] == 'inc_liq': # IncreaseLiquidity
                    res_tuple = pool_ranges.mint_token(
                        row['liquidity'], row['block'], row['tokenId'],
                    )
                else: # DecreaseLiquidity
                    res_tuple = pool_ranges.burn_token(
                        row['liquidity'], row['block'], row['tokenId'],
                    )
                if res_tuple is not None:
                    owner, res0, res1, last_blk, gauged = res_tuple
                    if owner in BLACKLIST:
                        continue
                    if res0 < 0 and res1 < 0:
                        print(f'negative balance in averageTokenInLPNFT @ blk({last_blk}), owner({owner}), res0({res0}), res1({res1})')
                        raise
                    if df_price0 is not None:
                        wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_price0)
                        if wtd_px is not None:
                            _val = wtd_px / total_time * res0
                            twab_by_token[pool_ranges.token0][owner] += _val
                            twab_by_pool[pool_address][pool_ranges.token0] += _val
                    if df_price1 is not None:
                        wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_price1)
                        if wtd_px is not None:
                            _val = wtd_px / total_time * res1
                            twab_by_token[pool_ranges.token1][owner] += _val
                            twab_by_pool[pool_address][pool_ranges.token1] += _val
        for addr, res0, res1, last_blk, _ in pool_ranges.get_current_reserves(latest_blk_num=blk_before_to): # set last blk to from_blk
            if addr in BLACKLIST:
                continue
            if res0 < 0 and res1 < 0:
                print(f'negative balance in averageTokenInLPNFT @ blk({last_blk}), owner({addr}), res0({res0}), res1({res1})')
                raise
            if df_price0 is not None:
                wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=blk_before_to, df_price=df_price0)
                if wtd_px is not None:
                    _val = wtd_px / total_time * res0
                    twab_by_token[pool_ranges.token0][addr] += _val
                    twab_by_pool[pool_address][pool_ranges.token0] += _val
                _val_after_to_blk = df_price0.iloc[-1].px * time_weight_after_to_blk * res0
                twab_by_token[pool_ranges.token0][addr] += _val_after_to_blk
                twab_by_pool[pool_address][pool_ranges.token0] += _val_after_to_blk
            if df_price1 is not None:
                wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=blk_before_to, df_price=df_price1)
                if wtd_px is not None:
                    _val = wtd_px / total_time * res1
                    twab_by_token[pool_ranges.token1][addr] += _val
                    twab_by_pool[pool_address][pool_ranges.token1] += _val
                _val_after_to_blk = df_price1.iloc[-1].px * time_weight_after_to_blk * res1
                twab_by_token[pool_ranges.token1][addr] += _val_after_to_blk
                twab_by_pool[pool_address][pool_ranges.token1] += _val_after_to_blk
    return twab_by_token, twab_by_pool, pool_ranges_by_address

def averageCVRSTAKING(
        from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
        df_price_cvr, stk_balances=None
):
    total_time = to_ts - from_ts
    twab = defaultdict(int)
    if stk_balances is not None:
        for addr, bal in stk_balances.items():
            twab[addr] = bal[0] * df_price_cvr.iloc[0].px * (blk_ts_after_from - from_ts) / total_time
            stk_balances[addr][-1] = blk_after_from
    else:
        stk_balances = dict() # address -> [cvr_bal, last_blk]
    
    stk_df = []
    dp_events = get_events(contracts['cvr_staking_chef'].events.Deposit, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    # user/amount
    if len(dp_events['block']) > 0:
        df_dp = pd.DataFrame(dp_events)
        df_dp.rename(columns={'user': 'owner'}, inplace=True)
        df_dp['amount'] = df_dp['amount'].astype(float)
        stk_df.append(df_dp)
    wd_events = get_events(contracts['cvr_staking_chef'].events.Withdraw, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(wd_events['block']) > 0:
        df_wd = pd.DataFrame(wd_events)
        df_wd.rename(columns={'user': 'owner'}, inplace=True)
        df_wd['amount'] = -df_wd['amount'].astype(float)
        stk_df.append(df_wd)
    if len(stk_df) > 0:
        df_stk = pd.concat(stk_df)
        df_stk['ix'] = df_stk.apply(lambda row: (row.block<<20) + (row.transactionIndex<<10) + (row.logIndex), axis=1)
        df_stk = df_stk.sort_values('ix')
        for _, row in df_stk.iterrows():
            if row['owner'] in BLACKLIST:
                continue
            if row['owner'] in stk_balances:
                bal, last_blk = stk_balances[row['owner']]
                wtd_px = _get_wtd_price_btw_blk(from_blk=last_blk, to_blk=row['block'], df_price=df_price_cvr)
                if wtd_px is not None:
                    twab[row['owner']] += bal * wtd_px / total_time
                    stk_balances[row['owner']][0] += row['amount']
                    stk_balances[row['owner']][-1] = row.block                
            else:
                stk_balances[row['owner']] = [row['amount'], row.block]
    for addr, bal_and_blk in stk_balances.items():
        wtd_px = _get_wtd_price_btw_blk(from_blk=bal_and_blk[-1], to_blk=blk_before_to, df_price=df_price_cvr)
        if wtd_px is not None:
            twab[addr] += bal_and_blk[0] * wtd_px / total_time
        twab[addr] += bal_and_blk[0] * df_price_cvr.iloc[-1].px * (to_ts - blk_ts_before_to) / total_time
        stk_balances[addr][-1] = blk_before_to
    return twab, stk_balances

def averageWrappedTokenInWallet(
    token_contract,
    from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
    df_price, running_balances
):
    total_time = to_ts - from_ts
    weighted_avg = defaultdict(int)
    
    for addr, (bal, _) in running_balances.items():
        weighted_avg[addr] += df_price.iloc[0].px * (blk_ts_after_from - from_ts) / total_time * bal
        running_balances[addr][1] = blk_after_from
    if NULL_ADDR not in running_balances:
        running_balances[NULL_ADDR] = [0, blk_after_from]
    
    df_xfers = []
    dp_events = get_events(token_contract.events.Deposit, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(dp_events) > 0:
        df_dp = pd.DataFrame(dp_events)
        df_dp['event'] = 'dp'
        df_dp['from'] = NULL_ADDR
        df_xfers.append(df_dp)
    wd_events = get_events(token_contract.events.Withdrawal, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(wd_events) > 0:
        df_wd = pd.DataFrame(wd_events)
        df_wd['event'] = 'wd'
        df_wd['to'] = NULL_ADDR
        df_xfers.append(df_wd)
    xfer_events = get_events(token_contract.events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(xfer_events) > 0:
        df_xfer = pd.DataFrame(xfer_events)
        df_xfer['event'] = 'xfer'
        df_xfers.append(df_xfer)
    if len(df_xfers) > 0:
        df_xfers = pd.concat(df_xfers)
        df_xfers['ix'] = df_xfers.apply(lambda row: (int(row.block)<<20) + (int(row.transactionIndex)<<10) + int(row.logIndex), axis=1)
        df_xfers = df_xfers.sort_values('ix')
        df_xfers['value'] = df_xfers['value'].astype(float)
    
        for _, row in df_xfers.iterrows():
            from_addr, to_addr, amt, blk = row['from'], row['to'], row['value'], row['block']
            if to_addr in running_balances:
                bal, last_blk = running_balances[to_addr]
                wtd_px = _get_wtd_price_btw_blk(last_blk, blk, df_price)
                if wtd_px is not None:
                    weighted_avg[to_addr] += wtd_px / total_time * bal
                    running_balances[to_addr][0] += amt
                    running_balances[to_addr][1] = blk
            else:
                if to_addr not in BLACKLIST:
                    running_balances[to_addr] = [amt, blk]
            
            if from_addr in running_balances:
                bal, last_blk = running_balances[from_addr]
                wtd_px = _get_wtd_price_btw_blk(last_blk, blk, df_price)
                if wtd_px is not None:
                    weighted_avg[from_addr] += wtd_px / total_time * bal
                    running_balances[from_addr][0] -= amt
                    if from_addr not in BLACKLIST and running_balances[from_addr][0] < -1e9:
                        if running_balances[from_addr][0] <= -1e18:
                            print(f'{from_addr} has negative running balance on {token_contract.address}: {running_balances[from_addr][0]/1e18}, fix to 0')
                            print(row.to_dict())
                        running_balances[from_addr][0] = 0
                    running_balances[from_addr][1] = blk
            else:
                if from_addr not in BLACKLIST:
                    print(f'first xfer {amt} from non-zero address, balance fix to 0')
                    print(row.to_dict())
                    running_balances[from_addr] = [0, blk]
    for addr, (bal, last_blk) in running_balances.items():
        wtd_px = _get_wtd_price_btw_blk(last_blk, blk_before_to, df_price)
        if wtd_px is not None:
            weighted_avg[addr] += wtd_px / total_time * bal
        wtd_px = df_price.iloc[-1].px * (to_ts - blk_ts_before_to) / total_time * bal
        if addr not in BLACKLIST and wtd_px < -1e9:
            print('end of day average token in wallet', addr, running_balances[addr], df_price.iloc[-1].px, 'fix to 0')
            wtd_px = 0
        weighted_avg[addr] += wtd_px
        running_balances[addr][1] = blk_before_to
    return weighted_avg, running_balances

def averageTokenInWallet(
        token_contract, from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
        df_price, running_balances=None
    ):
    total_time = to_ts - from_ts
    twab = defaultdict(int)
    for addr, (bal, _) in running_balances.items():
        twab[addr] += df_price.iloc[0].px * (blk_ts_after_from - from_ts) / total_time * bal
        running_balances[addr][1] = blk_after_from
    if NULL_ADDR not in running_balances:
        running_balances[NULL_ADDR] = [0, blk_after_from]
    
    xfer_events = get_events(token_contract.events.Transfer, blk_after_from, blk_before_to)
    if len(xfer_events) == 0:
        return twab, running_balances
    df_xfers = pd.DataFrame(xfer_events).sort_values('block')
    from_field = 'from'
    to_field = 'to'
    amt_field = 'value'
    df_xfers[amt_field] = df_xfers[amt_field].astype(float)
    for _, row in df_xfers.iterrows():
        wtd_px = None
        if row[to_field] in running_balances:
            bal, last_blk = running_balances[row[to_field]]
            wtd_px = _get_wtd_price_btw_blk(last_blk, row.block, df_price)
            if wtd_px is not None:
                twab[row[to_field]] += wtd_px / total_time * bal
                running_balances[row[to_field]][0] += row[amt_field]
                running_balances[row[to_field]][1] = row.block
        else:
            if row[to_field] not in BLACKLIST:
                running_balances[row[to_field]] = [row[amt_field], row.block]
        
        if row[from_field] in running_balances:
            bal, last_blk = running_balances[row[from_field]]
            wtd_px = _get_wtd_price_btw_blk(last_blk, row.block, df_price) if wtd_px is None else wtd_px
            if wtd_px is not None:
                twab[row[from_field]] += wtd_px / total_time * bal
                running_balances[row[from_field]][0] -= row[amt_field]
                if row[from_field] not in BLACKLIST and running_balances[row[from_field]][0] < -1e9:
                    if running_balances[row[from_field]][0] <= -1e18:
                        print(f'{row[from_field]} running balance {running_balances[row[from_field]][0]/1e18}, fix to 0')
                        print(row.to_dict())
                    running_balances[row[from_field]][0] = 0
                running_balances[row[from_field]][1] = row.block
        else:
            if row[from_field] not in BLACKLIST:
                print(f'first xfer {row[amt_field]} from non-zero address, balance fix to 0')
                print(row.to_dict())
            running_balances[row[from_field]] = [-row[amt_field], row.block]
    for addr, (bal, last_blk) in running_balances.items():
        wtd_px = _get_wtd_price_btw_blk(last_blk, blk_before_to, df_price)
        if wtd_px is not None:
            twab[addr] += wtd_px / total_time * bal
        wtd_px = df_price.iloc[-1].px * (to_ts - blk_ts_before_to) / total_time * bal
        if addr not in BLACKLIST and wtd_px < -1e9:
            print('end of day average token in wallet', addr, running_balances[addr], df_price.iloc[-1].px, 'fix to 0')
            wtd_px = 0
        twab[addr] += wtd_px
        running_balances[addr][1] = blk_before_to
    return twab, running_balances

def get_ve_token_by_owner(ve_contract, blk_num, owner):
    for _ in range(10):
        try:
            nft_len = ve_contract.functions.balanceOf(owner).call(block_identifier=blk_num)
            res = {}
            for i in range(nft_len):
                tid_from_owner = ve_contract.functions.tokenOfOwnerByIndex(owner, i).call(block_identifier=blk_num)
                res[tid_from_owner] = ve_contract.functions.getLockedAmount(tid_from_owner).call(block_identifier=blk_num)
            return res
        except Exception as e:
            print(f'get_ve_token_by_owner failed: {e}')
            time.sleep(1)
    raise

def update_stack_vaults(vault_factory_contract, from_blk=1, to_blk='latest', stack_vaults=None, w3_chain=W3):
    if to_blk == 'latest':
        to_blk = w3_chain.eth.block_number
    if stack_vaults is None:
        stack_vaults = {}
    vault_created_events = get_events(vault_factory_contract.events.VaultCreated, from_blk, to_blk)
    print(f"found {len(vault_created_events['block'])} vault")
    for i in range(len(vault_created_events['block'])):
        stack_vaults[vault_created_events['vault'][i]] = vault_created_events['collateralToken'][i]
    return stack_vaults

class MoreBalances:
    SMORE_LAST_BLK = 39909
    SMORE_SUPPLY = 0
    SMORE_ASSETS = 0

    @classmethod
    def set_smore_supply_and_assets(cls, latest_blk):
        if latest_blk < cls.SMORE_LAST_BLK:
            return
        else:
            smore_supply = contracts['smore'].functions.totalSupply().call(block_identifier=latest_blk)
            smore_assets = contracts['smore'].functions.totalAssets().call(block_identifier=latest_blk)
        cls.SMORE_SUPPLY = smore_supply
        cls.SMORE_ASSETS = smore_assets
        cls.SMORE_LAST_BLK = latest_blk
    
    @classmethod
    def assets_per_share(cls):
        return cls.SMORE_ASSETS / cls.SMORE_SUPPLY

    def __init__(self):
        self.more_bal = defaultdict(int)
        self.more_bal[''] = 0
        self.smore_shr = 0
        self.collat_bal_dict = defaultdict(int)
        self.last_blk = 0
    
    def update_to_blk(self, to_blk, df_price_more, df_token_price_by_blk, total_time):
        wtd_px = _get_wtd_price_btw_blk(from_blk=self.last_blk, to_blk=to_blk, df_price=df_price_more)
        if wtd_px is not None:
            wtd_px /= total_time
            weighted_more_to_add = int(self.more_bal[''] * wtd_px)
            weighted_brw_to_add = dict()
            weighted_collat_to_add = dict()
            for col_addr, col_bal in self.collat_bal_dict.items():
                if col_bal > 0:
                    if col_addr not in weighted_collat_to_add:
                        weighted_collat_to_add[col_addr] = defaultdict(int)
                        weighted_brw_to_add[col_addr] = defaultdict(int)
                    if col_addr == contracts['pta'].address:
                        df_collat_price = df_token_price_by_blk[contracts['arcusd'].address].copy()
                    elif col_addr == REETH_ADDRESS:
                        df_collat_price = df_token_price_by_blk[contracts['wreeth'].address].copy()
                    else:
                        df_collat_price = df_token_price_by_blk[col_addr].copy()
                        df_collat_price.px *= (df_collat_price['index'])
                    wtd_col = _get_wtd_price_btw_blk(from_blk=self.last_blk, to_blk=to_blk, df_price=df_collat_price)
                    if wtd_col is not None:
                        twab_col = col_bal * wtd_col / total_time
                        weighted_collat_to_add[col_addr] = twab_col
                        weighted_brw_to_add[col_addr] = self.more_bal[col_addr] * wtd_px / total_time
            weighted_smore_to_add = int(self.assets_per_share() * self.smore_shr * wtd_px) if self.smore_shr > 0 else 0
            return weighted_more_to_add, weighted_collat_to_add, weighted_smore_to_add, weighted_brw_to_add
        else:
            return 0, {}, 0, {}

def averageMoreDebt3(
    from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
    df_token_price_by_blk, more_balances, df_rebase_by_address
):
    total_time = to_ts - from_ts
    weighted_smore = defaultdict(int)
    weighted_more = defaultdict(int)
    weighted_collat = dict()
    df_price_more = df_token_price_by_blk[contracts['more'].address]
    twab_by_vault = dict()

    MoreBalances.set_smore_supply_and_assets(blk_after_from-1)

    if len(more_balances) > 0:
        time_weights = (blk_ts_after_from - from_ts) / total_time
        for addr, mbal in more_balances.items():
            weighted_more[addr] = int(mbal.more_bal[''] * df_price_more.iloc[0].px * time_weights)
            for col_addr, col_bal in mbal.collat_bal_dict.items():
                if col_addr == contracts['pta'].address:
                    px_at_from_blk = df_token_price_by_blk[contracts['arcusd'].address].iloc[0].px
                elif col_addr == REETH_ADDRESS:
                    px_at_from_blk = df_token_price_by_blk[contracts['wreeth'].address].iloc[0].px
                else:
                    px_at_from_blk = df_token_price_by_blk[col_addr].iloc[0].px * (df_token_price_by_blk[col_addr].iloc[0]['index'])
                if col_addr not in weighted_collat:
                    weighted_collat[col_addr] = defaultdict(int)
                twab_col = int(col_bal * px_at_from_blk * time_weights)
                weighted_collat[col_addr][addr] += twab_col
                if col_addr not in twab_by_vault:
                    twab_by_vault[col_addr] = [0, 0]
                twab_by_vault[col_addr][0] += twab_col
                twab_by_vault[col_addr][1] += mbal.more_bal[col_addr] * df_price_more.iloc[0].px * time_weights
            weighted_smore[addr] = int(MoreBalances.assets_per_share() * mbal.smore_shr * df_price_more.iloc[0].px * time_weights) if mbal.smore_shr > 0 else 0
            more_balances[addr].last_blk = blk_after_from
    
    df_more = []
    xfer_events = get_events(contracts['more'].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(xfer_events) > 0:
        df_xfer = pd.DataFrame(xfer_events)
        df_xfer.rename(columns={'value':'assets'}, inplace=True)
        df_xfer['assets'] = df_xfer['assets'].astype(float)
        df_xfer['shares'] = 0
        df_xfer['col_addr'] = ""
        df_xfer['index'] = 1
        df_more.append(df_xfer)
    smore_xfer_events = get_events(contracts['smore'].events.Transfer, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'])
    if len(smore_xfer_events['block']) > 0:
        df_sxfer = pd.DataFrame(smore_xfer_events)
        df_sxfer.rename(columns={'value':'shares'}, inplace=True)
        df_sxfer['shares'] = df_sxfer['shares'].astype(float)
        df_sxfer['assets'] = 0
        df_sxfer['col_addr'] = ""
        df_more.append(df_sxfer)

    vault_created_events = get_events(contracts['vault_factory'].events.VaultCreated, 1, blk_before_to)
    stack_vaults = {vault_created_events['vault'][i]: vault_created_events['collateralToken'][i] for i in range(len(vault_created_events['block']))}
    collat_to_vault = {}
    for vault, vault_collateral_token in stack_vaults.items():
        collat_to_vault[vault_collateral_token] = vault
        df_rebase = df_rebase_by_address[vault_collateral_token] if vault_collateral_token in df_rebase_by_address else None
        _vault_events = W3.eth.contract(vault, abi=abis['stack_vault']).events

        collat_db_event = get_events(_vault_events.CollateralDeposited, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'share':None, 'from':None})
        if len(collat_db_event['block']) > 0:
            df_db = pd.DataFrame(collat_db_event)
            df_db['from'] = 'collateral'
            df_db.rename(columns={'amount':'assets'}, inplace=True)
            df_db['assets'] = df_db['assets'].astype(float)
            df_db['shares'] = 0
            df_db['col_addr'] = vault_collateral_token
            if df_rebase is not None:
                df_db = pd.merge_asof(df_db, df_rebase[['index', 'block']], on='block')
            else:
                df_db['index'] = 1
            df_more.append(df_db)
        collat_wd_event = get_events(_vault_events.CollateralWithdrawn, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'share':None})
        if len(collat_wd_event['block']) > 0:
            df_wd = pd.DataFrame(collat_wd_event)
            df_wd['to'] = df_wd['from']
            df_wd['from'] = 'collateral'
            df_wd.rename(columns={'amount':'assets'}, inplace=True)
            df_wd['assets'] = -df_wd['assets'].astype(float)
            df_wd['shares'] = 0
            df_wd['col_addr'] = vault_collateral_token
            if df_rebase is not None:
                df_wd = pd.merge_asof(df_wd, df_rebase[['index', 'block']], on='block')
            else:
                df_wd['index'] = 1
            df_more.append(df_wd)
        borrow_event = get_events(_vault_events.Borrowed, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'to':None, 'share':None})
        if len(borrow_event['block']) > 0:
            df_brw = pd.DataFrame(borrow_event)
            df_brw['from'] = 'borrow'
            df_brw.rename(columns={'amount':'assets', 'borrower':'to'}, inplace=True)
            df_brw['assets'] = df_brw['assets'].astype(float)
            df_brw['shares'] = 0
            df_brw['col_addr'] = vault_collateral_token
            df_more.append(df_brw)
        repay_event = get_events(_vault_events.Repaid, blk_after_from, blk_before_to, extra_args=['transactionIndex', 'logIndex'], blacklist_args={'borrower':None, 'share':None}
        )# 'to', 'amount'
        if len(repay_event['block']) > 0:
            df_rpd = pd.DataFrame(repay_event)
            df_rpd['from'] = 'borrow'
            df_rpd.rename(columns={'amount':'assets'}, inplace=True)
            df_rpd['assets'] = -df_rpd['assets'].astype(float)
            df_rpd['shares'] = 0
            df_rpd['col_addr'] = vault_collateral_token
            df_more.append(df_rpd)
    if len(df_more) > 0:
        df_more = pd.concat(df_more)
        df_more['block'] = df_more['block'].astype('int64')
        df_more['ix'] = df_more.apply(lambda row: (int(row.block)<<20) + (int(row.transactionIndex)<<10) + int(row.logIndex), axis=1)
        df_more = df_more.sort_values('ix')
        df_more['index'] = df_more['index'].ffill() if 'index' in df_more else 1
        for _, row in df_more.iterrows():
            if row['from'] in more_balances:
                mbal = more_balances[row['from']]
                more_to_add, collat_to_add, smore_to_add, brw_to_add = mbal.update_to_blk(to_blk=row['block'], df_price_more=df_price_more, df_token_price_by_blk=df_token_price_by_blk, total_time=total_time)
                weighted_more[row['from']] += more_to_add
                for col_addr, col_bal in collat_to_add.items():
                    if col_addr not in weighted_collat:
                        weighted_collat[col_addr] = defaultdict(int)
                    weighted_collat[col_addr][row['from']] += col_bal
                    if col_addr not in twab_by_vault:
                        twab_by_vault[col_addr] = [0, 0]
                    twab_by_vault[col_addr][0] += col_bal
                    twab_by_vault[col_addr][1] += brw_to_add[col_addr]
                weighted_smore[row['from']] += smore_to_add
            if row['to'] in more_balances:
                mbal = more_balances[row['to']]
                more_to_add, collat_to_add, smore_to_add, brw_to_add = mbal.update_to_blk(to_blk=row['block'], df_price_more=df_price_more, df_token_price_by_blk=df_token_price_by_blk, total_time=total_time)
                weighted_more[row['to']] += more_to_add
                for col_addr, col_bal in collat_to_add.items():
                    if col_addr not in weighted_collat:
                        weighted_collat[col_addr] = defaultdict(int)
                    weighted_collat[col_addr][row['to']] += col_bal
                    if col_addr not in twab_by_vault:
                        twab_by_vault[col_addr] = [0, 0]
                    twab_by_vault[col_addr][0] += col_bal
                    twab_by_vault[col_addr][1] += brw_to_add[col_addr]
                weighted_smore[row['to']] += smore_to_add
            if row['from'] == 'collateral':
                if row['to'] not in more_balances:
                    more_balances[row['to']] = MoreBalances()
                more_balances[row['to']].collat_bal_dict[row['col_addr']] += row['assets'] / row['index']
                more_balances[row['to']].last_blk = row.block
                if more_balances[row['to']].collat_bal_dict[row['col_addr']] < 0:
                    more_balances[row['to']].collat_bal_dict[row['col_addr']] = 0
            elif row['from'] == 'borrow':
                if row['to'] not in more_balances:
                    more_balances[row['to']] = MoreBalances()
                more_balances[row['to']].more_bal[row['col_addr']] += row['assets']
                more_balances[row['to']].last_blk = row.block
                if more_balances[row['to']].more_bal[row['col_addr']] < 0:
                    more_balances[row['to']].more_bal[row['col_addr']] = 0
            else:
                if row['assets'] != 0:
                    if row['to'] == contracts['smore'].address:
                        MoreBalances.SMORE_ASSETS += row['assets']
                        MoreBalances.SMORE_LAST_BLK = row['block']
                    elif row['to'] != NULL_ADDR:
                        if row['to'] not in more_balances:
                            more_balances[row['to']] = MoreBalances()
                        more_balances[row['to']].more_bal[''] += row['assets']
                        more_balances[row['to']].last_blk = row.block
                    if row['from'] == contracts['smore'].address:
                        MoreBalances.SMORE_ASSETS -= row['assets']
                        MoreBalances.SMORE_LAST_BLK = row['block']
                    elif row['from'] != NULL_ADDR:
                        if row['from'] not in more_balances:
                            more_balances[row['from']] = MoreBalances()
                        more_balances[row['from']].more_bal[''] -= row['assets']
                        more_balances[row['from']].last_blk = row.block
                        if more_balances[row['from']].more_bal[''] < 0:
                            more_balances[row['from']].more_bal[''] = 0
                else:
                    if row['from'] == NULL_ADDR:
                        MoreBalances.SMORE_SUPPLY += row['shares']
                        MoreBalances.SMORE_LAST_BLK = row['block']
                    else:
                        if row['from'] not in more_balances:
                            print(f'transfer smore({row["shares"]}) from an empty address: {row["from"]}')
                            print(f'events: {row.to_dict()}')
                        else:
                            more_balances[row['from']].smore_shr -= row['shares']
                            more_balances[row['from']].last_blk = row.block
                            if more_balances[row['from']].smore_shr < 0:
                                more_balances[row['from']].smore_shr = 0
                    if row['to'] == NULL_ADDR:
                        MoreBalances.SMORE_SUPPLY -= row['shares']
                        MoreBalances.SMORE_LAST_BLK = row['block']
                        if MoreBalances.SMORE_SUPPLY < 0:
                            print(f'negative smore total supply')
                            print(f'events: {row.to_dict()}')
                            MoreBalances.SMORE_SUPPLY = 0
                    else:
                        if row['to'] not in more_balances:
                            more_balances[row['to']] = MoreBalances()
                        more_balances[row['to']].smore_shr += row['shares']
                        more_balances[row['to']].last_blk = row.block
    time_weights = (to_ts - blk_ts_before_to) / total_time
    for addr, mbal in more_balances.items():
        more_to_add, collat_to_add, smore_to_add, brw_to_add = mbal.update_to_blk(to_blk=blk_before_to, df_price_more=df_price_more, df_token_price_by_blk=df_token_price_by_blk, total_time=total_time)
        weighted_more[addr] += more_to_add
        weighted_smore[addr] += smore_to_add
        for col_addr, col_bal in mbal.collat_bal_dict.items():
            if col_addr not in weighted_collat:
                weighted_collat[col_addr] = defaultdict(int)
            if col_addr not in twab_by_vault:
                twab_by_vault[col_addr] = [0, 0]
            if col_addr in collat_to_add:
                weighted_collat[col_addr][addr] += collat_to_add[col_addr]
                twab_by_vault[col_addr][0] += collat_to_add[col_addr]
                twab_by_vault[col_addr][1] += brw_to_add[col_addr]
            if col_addr == contracts['pta'].address:
                px_at_to_blk = df_token_price_by_blk[contracts['arcusd'].address].iloc[-1].px
            elif col_addr == REETH_ADDRESS:
                px_at_to_blk = df_token_price_by_blk[contracts['wreeth'].address].iloc[-1].px
            else:
                px_at_to_blk = df_token_price_by_blk[col_addr].iloc[-1].px * (df_token_price_by_blk[col_addr].iloc[-1]['index'])
            val_after_to_blk = int(col_bal * px_at_to_blk * time_weights)
            weighted_collat[col_addr][addr] += val_after_to_blk
            twab_by_vault[col_addr][0] += val_after_to_blk
            twab_by_vault[col_addr][1] += mbal.more_bal[col_addr] * df_price_more.iloc[-1].px * time_weights
        mbal.last_blk = blk_before_to
    return weighted_more, weighted_smore, weighted_collat, twab_by_vault, more_balances, stack_vaults, collat_to_vault

def get_daily_snapshot_for_ve(points_owner_addresses, snap_ts, ve_contract, df_price):
    wtd_ve_value = {}
    blk_before_snap, _, _, _, _ = _get_block_num_from_ts(snap_ts, W3)
    snap_ix = np.searchsorted(df_price.index, blk_before_snap, side='right')
    snap_px = (df_price.iloc[snap_ix].px if snap_ix < len(df_price) else df_price.iloc[-1].px)
    for addr in points_owner_addresses:
        if type(addr) == str and addr != '' and addr != NULL_ADDR:
            tid_to_amt = get_ve_token_by_owner(ve_contract, blk_before_snap, addr)
            wtd_ve_value[addr] = sum([amt for _, amt in tid_to_amt.items()]) * snap_px
    return wtd_ve_value

def get_daily_snapshot_for_reeth(points_owner_addresses, snap_ts, df_eth_price):
    wtd_reeth_value = {}
    blk_before_snap, _, _, _, _ = _get_block_num_from_ts(snap_ts, W3)
    snap_ix = np.searchsorted(df_eth_price.index, blk_before_snap, side='right')
    snap_px = (df_eth_price.iloc[snap_ix].px if snap_ix < len(df_eth_price) else df_eth_price.iloc[-1].px)
    for addr in points_owner_addresses:
        if type(addr) == str and addr != '' and addr != NULL_ADDR:
            wtd_reeth_value[addr] = W3.eth.get_balance(addr, block_identifier=blk_before_snap) * snap_px
    return wtd_reeth_value

def get_refering():
    while True:
        try:
            res = requests.get('https://api.tangible.store/points/wallets').json()
            if 'dataWallets' not in res:
                print(res)
                print('no dataWallets, retrying')
                continue
            # json.dump(res, open(f'refering_{dt.datetime.now(dt.timezone.utc).date()}.json', 'w'))
            ref_res = {}
            for user_info in res['dataWallets']:
                if 'usedWalletAddress' not in user_info:
                    print(user_info)
                if user_info['usedWalletAddress'] != '':
                    ref_res[user_info['walletAddress']] = user_info['usedWalletAddress']
            ref_res = {user_info['walletAddress']:user_info['usedWalletAddress'] for user_info in res['dataWallets'] if user_info['usedWalletAddress'] != ''}
            print(ref_res)
            return ref_res
        except Exception as e:
            print(e)
            print('retrying')
        time.sleep(10)

def refering_points(pts, ref):
    # ref = get_refering()
    weighted_avg = pts.to_dict()
    refer_pts = defaultdict(int)
    for addr in list(weighted_avg.keys()):
        val = weighted_avg[addr]
        from_addr = addr
        while from_addr in ref and ref[from_addr] != NULL_ADDR:
            # print(f'refering transfer {val} from {from_addr} to {ref[from_addr]}')
            val /= 9
            if val < 0.01:
                break
            refer_pts[ref[from_addr]] += val
            from_addr = ref[from_addr]
    for addr in refer_pts:
        if addr in weighted_avg:
            weighted_avg[addr] = weighted_avg[addr] + refer_pts[addr]
        else:
            weighted_avg[addr] = refer_pts[addr]
    return {key:int(val) for key, val in weighted_avg.items() if int(val) > 0}

def _add_column(wtd_val_dict, df_summary, col_name):
    if len(wtd_val_dict) > 0:
        df_summary = df_summary.merge(pd.Series(wtd_val_dict).astype(float).rename(col_name), how='outer', left_index=True, right_index=True)
    else:
        df_summary[col_name] = 0
    return df_summary

def load_from_pickle(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        print(f'{file_name} not found')
        return {}
        
def day_by_day_summary(root_path='./', start_date=None, end_date=None):
    start_date = '2024-05-16' if start_date is None else start_date
    end_date = str((dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).date()) if end_date is None else end_date
    PearlV2Pool.set_nft_manager(contracts['pearl_nft_manager'])
    if os.path.exists(root_path+f'daily_data/pearl_pools/classdata_{start_date}.pkl'):
        pearl_pool_meta = pickle.load(open(root_path+f'daily_data/pearl_pools/classdata_{start_date}.pkl', 'rb'))
        for key, item in pearl_pool_meta.items():
            setattr(PearlV2Pool, key, item)

    visited = set()
    report_path = root_path+f'daily_reports/rwa'
    if not os.path.exists(report_path):
        os.makedirs(report_path)

    pool_path = root_path+f'daily_data/pearl_pools'
    if not os.path.exists(pool_path):
        os.makedirs(pool_path)
    pearl_pools = load_from_pickle(pool_path+f'/{start_date}.pkl')

    running_alm_path = root_path+f'daily_data/alm_positions'    
    if not os.path.exists(running_alm_path):
        os.makedirs(running_alm_path)
    running_alm_position = load_from_pickle(running_alm_path+f'/{start_date}.pkl')
    
    gauge_path = root_path+f'daily_data/gauges'
    if not os.path.exists(gauge_path):
        os.makedirs(gauge_path)
    gauges_by_pool = load_from_pickle(gauge_path+f'/{start_date}.pkl')

    running_staked_cvr_path = root_path+f'daily_data/staked_cvr'
    if not os.path.exists(running_staked_cvr_path):
        os.makedirs(running_staked_cvr_path)
    stk_balances = load_from_pickle(running_staked_cvr_path+f'/{start_date}.pkl')

    running_more_path = root_path+f'daily_data/more_balances'
    if not os.path.exists(running_more_path):
        os.makedirs(running_more_path)
    more_balances = load_from_pickle(running_more_path+f'/{start_date}.pkl')

    last_day_rebase_path = root_path+f'daily_data/last_day_rebase'
    if not os.path.exists(last_day_rebase_path):
        os.makedirs(last_day_rebase_path)
    last_shares_by_address = load_from_pickle(last_day_rebase_path+f'/{start_date}.pkl')

    running_token_balances_path = root_path+f'daily_data/token_balances'
    if not os.path.exists(running_token_balances_path):
        os.makedirs(running_token_balances_path)
    tokens_balances = load_from_pickle(running_token_balances_path+f'/{start_date}.pkl')

    running_ve_balances_path = root_path+f'daily_data/ve_balances'
    if not os.path.exists(running_ve_balances_path):
        os.makedirs(running_ve_balances_path)
    ve_balances = load_from_pickle(running_ve_balances_path+f'/{start_date}.pkl')

    running_tid_to_locked_balances_path = root_path+f'daily_data/tid_to_locked_balances'
    if not os.path.exists(running_tid_to_locked_balances_path):
        os.makedirs(running_tid_to_locked_balances_path)
    running_tid_to_locked_balances = load_from_pickle(running_tid_to_locked_balances_path+f'/{start_date}.pkl')

    date_to_snap_ts_path = root_path+f'date_to_snap_ts.pkl'
    date_snap_ts_dict = load_from_pickle(date_to_snap_ts_path)

    latest_ref = get_refering()

    for dt_date in pd.date_range(start_date, end_date):
        date_str = dt_date.date()
        season = day_to_season(date_str)
        
        total_token, days_in_season, _, _ = TOKEN_BY_SEASON[season]
        daily_rwa_token = total_token / days_in_season
        next_date_str = (dt_date + dt.timedelta(days=1)).date()
        
        print(date_str)
        df_summary = pd.DataFrame(data={'tvl_pts':0}, index=[''])
        from_ts = int(dt_date.timestamp())
        to_ts = from_ts + 60*60*24
        
        _, _, blk_after_from, blk_ts_after_from, from_ts = _get_block_num_from_ts(from_ts, W3)
        blk_before_to, blk_ts_before_to, _, _, to_ts = _get_block_num_from_ts(to_ts, W3)
        print(f'from ts:{from_ts}, from blk ts: {blk_ts_after_from}({blk_after_from}), to blk ts:{blk_ts_before_to}({blk_before_to})')

        if blk_after_from >= blk_before_to:
            continue
        
        if len(pearl_pools) > 0:
            for key, pool in pearl_pools.items():
                pool.contract = W3.eth.contract(address=pool.contract, abi=abis['pearlv2_pool'])
                TupleToPearlPool[(pool.token0, pool.token1, pool.fee)] = pool
        _get_new_pearl_pools(from_blk=blk_after_from, to_blk=blk_before_to, pools_by_address=pearl_pools)
        df_swap_by_address, df_pool_price_by_address = getPearlPoolSwap(blk_after_from, blk_before_to, pearl_pools, W3)

        df_token_price_by_blk = {}
        df_rebase_by_address = {}
        for token_address in ROUTE_TO_USTB.keys():
            if token_address not in address_to_symbol:
                continue
            df_token_price_by_blk[token_address], df_rebase = update_df_price_by_token(contracts[address_to_symbol[token_address]], blk_after_from, blk_before_to, df_pool_price_by_address, w3_chain=W3)
            if address_to_symbol[token_address] in REBASE_TOKENS:
                df_rebase_by_address[token_address] = df_rebase

        print('checking alm')
        twab_in_alms_by_token, twab_in_alms_by_pool = averageTokenInALM(
            from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
            df_token_price_by_blk, df_swap_by_address, running_positions=running_alm_position, w3_chain=W3
        )
        pickle.dump(running_alm_position, open(running_alm_path+f'/{next_date_str}.pkl', 'wb'))
        for token_addr, twab_in_alms in twab_in_alms_by_token.items():
            df_summary = _add_column(twab_in_alms, df_summary, f'{address_to_symbol[token_addr]}_in_alm')

        get_gauge_contracts(contracts['voter'], None, blk_after_from, blk_before_to, gauges_by_pool, W3)
        pickle.dump(gauges_by_pool, open(gauge_path+f'/{next_date_str}.pkl', 'wb'))
        
        print('checking lp_farm')
        twab_in_nft_by_token, twab_in_nft_by_pool, _ = averageTokenInLPNFT(
            from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
            df_swap_by_address, df_token_price_by_blk, pool_ranges_by_address=pearl_pools, tokens_to_track=None, gauge_by_address=gauges_by_pool
        )
        # print(twab_in_nft_by_pool)

        for token_addr, twab_in_nfts in twab_in_nft_by_token.items():
            if token_addr not in address_to_symbol:
                if token_addr != '0x8D7dd0C2FbfAF1007C733882Cd0ccAdEFFf275D2':
                    print(token_addr, 'not in address_to_symbol')
                continue
            df_summary = _add_column(twab_in_nfts, df_summary, f'{address_to_symbol[token_addr]}_in_nft')

        print('checking cvr balances')
        if 'cvr' not in tokens_balances:
            tokens_balances['cvr'] = {}
        cvr_twab_in_wallet, _ = averageTokenInWallet(
            contracts['cvr'], from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['cvr'].address],
            running_balances=tokens_balances['cvr']
        )
        df_summary = _add_column(cvr_twab_in_wallet, df_summary, 'cvr')
        
        print('checking cvr_staking')
        scvr_twab, _ = averageCVRSTAKING(
            from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['cvr'].address], stk_balances=stk_balances
        )
        pickle.dump(stk_balances, open(running_staked_cvr_path+f'/{next_date_str}.pkl', 'wb'))
        df_summary = _add_column(scvr_twab, df_summary, 'scvr')

        print('checking more token')
        twab_more, twab_smore, twab_collat_by_token, twab_by_vault, _, _ , collat_to_vault = averageMoreDebt3(
            from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
            df_token_price_by_blk, more_balances=more_balances, df_rebase_by_address=df_rebase_by_address
        )

        pickle.dump(more_balances, open(running_more_path+f'/{next_date_str}.pkl', 'wb'))
        df_summary = _add_column(twab_smore, df_summary, 'smore')
        df_summary = _add_column(twab_more, df_summary, 'more')
        for collat_addr, twab_collat in twab_collat_by_token.items():
            df_summary = _add_column(twab_collat, df_summary, f'{address_to_symbol[collat_addr]}_in_vault')
        if 'smore' in df_summary.index:
            raise
        if 'collat' in df_summary.index:
            raise
        
        print('checking rebase token')
        twab_rebase_by_token, _ = getAllRebaseEvents(
            from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to,
            df_token_price_by_blk, df_rebase_by_address=df_rebase_by_address, last_shares_by_address=last_shares_by_address
        )
        pickle.dump(last_shares_by_address, open(f'./daily_data/last_day_rebase/{next_date_str}.pkl', 'wb'))
        for token_symbol, twab_rebase in twab_rebase_by_token.items():
            df_summary = _add_column(twab_rebase, df_summary, token_symbol)
        
        print('checking swap fee')
        dfs_sender_to_fee = []
        for pool_addr, df_swap in df_swap_by_address.items():
            df_price0 = df_token_price_by_blk[pearl_pools[pool_addr].token0]
            df_price0['block'] = df_price0.index+1
            df = pd.merge_asof(df_swap[['block', 'recipient', 'amount0']], df_price0, on='block')
            df['value'] = abs(df['amount0']) * df['px'] * pearl_pools[pool_addr].fee / 1e6
            dfs_sender_to_fee.append(df)
        if len(dfs_sender_to_fee) > 0:
            df_swap_points = pd.concat(dfs_sender_to_fee).groupby('recipient')['value'].sum() / 1e18
            _swp_pcnt = min(SWP_PCNT_CAP, SWP_PTS_MULTIPLIER * df_swap_points.sum() / daily_rwa_token / df_token_price_by_blk[contracts['rwa'].address].px.iloc[-1])
            df_swap_points = df_swap_points / df_swap_points.sum()
            if len(df_summary) > 0:
                df_summary = df_summary.merge(df_swap_points.rename('swap_fee'), how='outer', left_index=True, right_index=True)
            else:
                df_summary['swap_fee'] = df_swap_points
        else:
            df_summary['swap_fee'] = 0
            _swp_pcnt = 0

        print('checking wreeth balances')
        if 'wreeth' not in tokens_balances:
            tokens_balances['wreeth'] = {}
        twab_wreeth, _ = averageWrappedTokenInWallet(
            contracts['wreeth'], from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['wreeth'].address],
            running_balances=tokens_balances['wreeth']
        )
        df_summary = _add_column(twab_wreeth, df_summary, 'wreeth')
        print('checking pta balances')
        if 'pta' not in tokens_balances:
            tokens_balances['pta'] = {}
        twab_pta, _ = averageTokenInWallet(
            contracts['pta'], from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['arcusd'].address],
            running_balances=tokens_balances['pta']
        )
        df_summary = _add_column(twab_pta, df_summary, 'pta')
        print('checking rwa balances')
        if 'rwa' not in tokens_balances:
            tokens_balances['rwa'] = {}
        twab_rwa, _ = averageTokenInWallet(
            contracts['rwa'], from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['rwa'].address],
            running_balances=tokens_balances['rwa']
        )
        df_summary = _add_column(twab_rwa, df_summary, 'rwa')
        print('checking pearl balances')
        if 'pearl' not in tokens_balances:
            tokens_balances['pearl'] = {}
        twab_pearl, _ = averageTokenInWallet(
            contracts['pearl'], from_ts, to_ts, blk_after_from, blk_before_to, blk_ts_after_from, blk_ts_before_to, df_token_price_by_blk[contracts['pearl'].address],
            running_balances=tokens_balances['pearl']
        )
        df_summary = _add_column(twab_pearl, df_summary, 'pearl')
 
        pickle.dump(ve_balances, open(running_ve_balances_path+f'/{next_date_str}.pkl', 'wb'))
        pickle.dump(tokens_balances, open(running_token_balances_path+f'/{next_date_str}.pkl', 'wb'))
        pickle.dump(running_tid_to_locked_balances, open(running_tid_to_locked_balances_path+f'/{next_date_str}.pkl', 'wb'))

        print('saving metadata to local')
        if len(pearl_pools) > 0:
            for key, pool in pearl_pools.items():
                pool.contract = pool.contract.address
            pickle.dump(pearl_pools, open(pool_path+f'/{next_date_str}.pkl', 'wb'))
            PearlV2Pool.save_classdata_to_local(next_date_str)
        for addr in BLACKLIST:
            if addr in df_summary.index:
                df_summary = df_summary.drop(addr)
        for addr in df_summary.index:
            if (type(addr) == str) and addr != '' and addr not in visited and addr not in BLACKLIST and isContract(addr):
                print(f'found new smart contract: {addr}')
                visited.add(addr)
        if len(df_summary) > 0:
            df_summary = df_summary.infer_objects(copy=False).fillna(0)
            if date_str not in date_snap_ts_dict:
                snap_ts = randint(from_ts, to_ts)
                date_snap_ts_dict[date_str] = snap_ts
                pickle.dump(date_snap_ts_dict, open(date_to_snap_ts_path, 'wb'))
            reeth_wtd_val = get_daily_snapshot_for_reeth(df_summary.index, date_snap_ts_dict[date_str], df_token_price_by_blk[contracts['wreeth'].address])
            df_summary = _add_column(reeth_wtd_val, df_summary, 'reeth')
            print('done checking reeth balances')

            verwa_wtd_val = get_daily_snapshot_for_ve(df_summary.index, date_snap_ts_dict[date_str], contracts['verwa'], df_token_price_by_blk[contracts['rwa'].address])
            vepearl_wtd_val = get_daily_snapshot_for_ve(df_summary.index, date_snap_ts_dict[date_str], contracts['vepearl'], df_token_price_by_blk[contracts['pearl'].address])
            df_summary = _add_column(verwa_wtd_val, df_summary, 'verwa')
            df_summary = _add_column(vepearl_wtd_val, df_summary, 'vepearl')
            print('done with ve')

            df_summary['raw_tvl'] = 0

            df_summary['tvl_pts'] += df_summary['verwa'] * 0.05
            df_summary['raw_tvl'] += df_summary['verwa']
            df_summary['tvl_pts'] += df_summary['vepearl'] * 0.05
            df_summary['raw_tvl'] += df_summary['vepearl']
            reeth_tvl = df_summary[[cname for cname in df_summary.columns if cname[:5] == 'reeth']].sum(axis=1)
            df_summary['tvl_pts'] += reeth_tvl
            df_summary['raw_tvl'] += reeth_tvl
            wreeth_val = df_summary[[cname for cname in df_summary.columns if cname[:6] == 'wreeth']].sum(axis=1)
            df_summary['tvl_pts'] += wreeth_val
            df_summary['raw_tvl'] += wreeth_val
            ustb_val = df_summary[[cname for cname in df_summary.columns if cname[:4] == 'ustb']].sum(axis=1)
            df_summary['tvl_pts'] += ustb_val
            df_summary['raw_tvl'] += ustb_val
            cvr_val = df_summary[[cname for cname in df_summary.columns if cname[:3] == 'cvr']].sum(axis=1)
            df_summary['tvl_pts'] += cvr_val
            df_summary['raw_tvl'] += cvr_val
            pearl_val = df_summary[[cname for cname in df_summary.columns if cname[:5] == 'pearl']].sum(axis=1)
            df_summary['tvl_pts'] += pearl_val
            df_summary['raw_tvl'] += pearl_val
            rwa_val = df_summary[[cname for cname in df_summary.columns if cname[:3] == 'rwa']].sum(axis=1)
            df_summary['tvl_pts'] += rwa_val
            df_summary['raw_tvl'] += rwa_val
            more_val = df_summary[[cname for cname in df_summary.columns if cname[:4] == 'more']].sum(axis=1)
            df_summary['tvl_pts'] += more_val
            df_summary['raw_tvl'] += more_val
            dai_val = df_summary[[cname for cname in df_summary.columns if cname[:3] == 'dai']].sum(axis=1)
            df_summary['tvl_pts'] += dai_val
            df_summary['raw_tvl'] += dai_val
            df_summary['tvl_pts'] += df_summary['scvr'] * 2.5
            df_summary['raw_tvl'] += df_summary['scvr']
            df_summary['tvl_pts'] += df_summary['smore'] * 2.5
            df_summary['raw_tvl'] += df_summary['smore']
            pta_val = df_summary[[cname for cname in df_summary.columns if cname[:3] == 'pta']].sum(axis=1)
            df_summary['tvl_pts'] += pta_val * 4
            df_summary['raw_tvl'] += pta_val
            arcusd_val = df_summary[[cname for cname in df_summary.columns if cname[:6] == 'arcusd']].sum(axis=1)
            df_summary['tvl_pts'] += arcusd_val * 4
            df_summary['raw_tvl'] += arcusd_val
            ukre_val = df_summary[[cname for cname in df_summary.columns if cname[:4] == 'ukre']].sum(axis=1)
            df_summary['tvl_pts'] += ukre_val * 5
            df_summary['raw_tvl'] += ukre_val
            df_summary['tvl_pts'] += df_summary[[cname for cname in df_summary.columns if cname[-3:] == 'alm']].sum(axis=1) * 2
            df_summary['tvl_pts'] += df_summary[[cname for cname in df_summary.columns if cname[-3:] == 'nft']].sum(axis=1) * 2
            # raw_twvl = df_summary['raw_tvl'].sum()
            df_summary['daily_norm_tvl'] = df_summary['tvl_pts'] / df_summary['tvl_pts'].sum()
            df_summary['daily_norm_swp'] = df_summary['swap_fee'] / df_summary['swap_fee'].sum() if df_summary['swap_fee'].sum() > 0 else 0
            df_summary['swap_pcnt'] = _swp_pcnt
            _tvl_pcnt = 1 - _swp_pcnt
            df_summary['daily_norm_pts'] = _tvl_pcnt * df_summary['daily_norm_tvl'] + _swp_pcnt * df_summary['daily_norm_swp']
            df_summary['raw_points'] = df_summary['daily_norm_pts'] * daily_rwa_token * 1000 * 0.9
            df_summary['points_ref_adj'] = refering_points(df_summary['raw_points'], latest_ref)
            df_summary.to_csv(report_path+f'/daily_summary_{date_str}.csv')

            total_rwd_val = daily_rwa_token * _tvl_pcnt * 0.9 * df_token_price_by_blk[contracts['rwa'].address].px.iloc[-1] * 365
            
            twab_by_pool = {}
            twab_by_pool['token'] = [1, 1]
            # print(twab_in_nft_by_pool)
            # print(twab_by_vault)
            for collat_addr, [twab_collat, twab_brw] in twab_by_vault.items():
                twab_by_pool[collat_to_vault[collat_addr]] = [
                    twab_collat * ((LEVERAGE_MULTIPLIER[collat_addr]) if collat_addr in LEVERAGE_MULTIPLIER else 1) + twab_brw,
                    twab_collat
                ]
            for pool_addr, twab_by_t in twab_in_nft_by_pool.items():
                twab_by_pool[pool_addr] = [0, 0]
                for _t in twab_by_t:
                    twab_by_pool[pool_addr][0] += twab_by_t[_t] * ((LEVERAGE_MULTIPLIER[_t] + 2) if _t in LEVERAGE_MULTIPLIER else 3)
                    twab_by_pool[pool_addr][1] += twab_by_t[_t]
            for pool_addr, twab_by_t in twab_in_alms_by_pool.items():
                twab_by_pool[pool_addr] = [0, 0]
                for _t in twab_by_t:
                    twab_by_pool[pool_addr][0] += twab_by_t[_t] * ((LEVERAGE_MULTIPLIER[_t] + 2) if _t in LEVERAGE_MULTIPLIER else 3)
                    twab_by_pool[pool_addr][1] += twab_by_t[_t]
            df_by_pool = pd.DataFrame.from_dict(twab_by_pool, orient='index', columns=['adj_twab', 'twab'])            
            df_by_pool.adj_twab = df_by_pool.adj_twab.astype(float)
            df_by_pool.twab = df_by_pool.twab.astype(float)

            apy_by_pool = df_by_pool.adj_twab / df_by_pool.twab * total_rwd_val / df_summary['tvl_pts'].sum() * 1e18
            apy_by_pool.to_csv(report_path+f'/apy_by_pool_{date_str}.csv')

if __name__ == '__main__':

    import sys
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('mode.chained_assignment', None)
    # get_refering()
    # raise
    
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    
    day_by_day_summary(start_date=start_date, end_date=end_date)

