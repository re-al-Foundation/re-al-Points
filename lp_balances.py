import bisect
from typing import Dict, List, Tuple
from numpy import sqrt
from collections import defaultdict
import pickle

class RangeInPx:
    # id -> owner address
    # id -> liquidity_tuple
    # mint id first, then add liquidity

    def __init__(self, lower_sqrt_px, upper_sqrt_px):
        self.virtual_multiplier0 = 1 / upper_sqrt_px
        self.virtual_multiplier1 = lower_sqrt_px
        self.max_multiplier0 = 1 / lower_sqrt_px - self.virtual_multiplier0
        self.max_multiplier1 = upper_sqrt_px - self.virtual_multiplier1
        self.owner_to_liquidity = dict() # Dict(address_or_tokenId:str_or_uint, List[liquidity:int, res0, res1, blk_num])


    def is_empty(self):
        return len(self.owner_to_liquidity) == 0

    
    def change_liquidity_in_range(self, token_id, delta_liquidity, sqrt_px, blk_num):
        if token_id in self.owner_to_liquidity:
            liquidity = self.owner_to_liquidity[token_id][0]
            liquidity += delta_liquidity
            if liquidity == 0:
                del self.owner_to_liquidity[token_id]
                return
                # del self.tokenId_to_owner[token_id]
        else:
            liquidity = delta_liquidity
        res0 = liquidity * (1 / sqrt_px - self.virtual_multiplier0)
        res1 = liquidity * (sqrt_px - self.virtual_multiplier1)
        self.owner_to_liquidity[token_id] = [liquidity, res0, res1, blk_num]
    
    def change_liquidity_lower(self, token_id, delta_liquidity, blk_num):
        # range < lower
        if token_id in self.owner_to_liquidity:
            liquidity = self.owner_to_liquidity[token_id][0]
            liquidity += delta_liquidity
            if liquidity == 0:
                del self.owner_to_liquidity[token_id]
                return
                # del self.tokenId_to_owner[token_id]
        else:
            liquidity = delta_liquidity
        self.owner_to_liquidity[token_id] = [liquidity, liquidity * self.max_multiplier0, 0, blk_num]

    def change_liquidity_upper(self, token_id, delta_liquidity, blk_num):
        # range < lower
        if token_id in self.owner_to_liquidity:
            liquidity = self.owner_to_liquidity[token_id][0]
            liquidity += delta_liquidity
            if liquidity == 0:
                del self.owner_to_liquidity[token_id]
                return
                # del self.tokenId_to_owner[token_id]
        else:
            liquidity = delta_liquidity
        self.owner_to_liquidity[token_id] = [liquidity, 0, liquidity * self.max_multiplier1, blk_num]
        
    def remove_liquidity(self, token_id, liquidity, blk_num):
        if token_id in self.owner_to_liquidity:
            if liquidity >= self.owner_to_liquidity[token_id][0]:
                self.owner_to_liquidity[token_id][1] = 0
                self.owner_to_liquidity[token_id][2] = 0
                self.owner_to_liquidity[token_id][0] = 0
                self.owner_to_liquidity[token_id][3] = blk_num
                # del self.tokenId_to_owner[token_id]
            else:
                _multiplier = (1 - liquidity / self.owner_to_liquidity[token_id][0])
                self.owner_to_liquidity[token_id][1] *= _multiplier
                self.owner_to_liquidity[token_id][2] *= _multiplier
                self.owner_to_liquidity[token_id][0] -= liquidity
                self.owner_to_liquidity[token_id][3] = blk_num
        else:
            print("Owner not found")
    
    def get_reserves_by_id(self, token_id):
        if token_id in self.owner_to_liquidity:
            last_amt0, last_amt1, last_blk_num = self.owner_to_liquidity[token_id][1], self.owner_to_liquidity[token_id][2], self.owner_to_liquidity[token_id][3]
            # owner = self.tokenId_to_owner[token_id]
            return [last_amt0, last_amt1, last_blk_num]
        return None
    
    def get_res_lower_range(self, blk_num):
        # tick <= lower
        last_res = [] # [(owner, res0, res1, last_blk_num)]
        for token_id, liq in self.owner_to_liquidity.items():
            # owner = self.tokenId_to_owner[token_id]
            last_res.append([token_id, liq[1], liq[2], liq[3]])
            liq[1] = liq[0] * self.max_multiplier0
            liq[2] = 0
            liq[3] = blk_num
        return last_res
    
    def get_res_upper_range(self, blk_num):
        # tick >= upper
        last_res = []
        for token_id, liq in self.owner_to_liquidity.items():
            # owner = self.tokenId_to_owner[token_id]
            last_res.append((token_id, liq[1], liq[2], liq[3]))
            liq[1] = 0
            liq[2] = liq[0] * self.max_multiplier1
            liq[3] = blk_num
        return last_res
    
    def price_change_in_range(self, latest_sqrt_px, blk_num):
        last_res = []
        for token_id, liq in self.owner_to_liquidity.items():
            # owner = self.tokenId_to_owner[token_id]
            last_res.append([token_id, liq[1], liq[2], liq[3]])
            liq[1] = liq[0] * (1 / latest_sqrt_px - self.virtual_multiplier0)
            liq[2] = liq[0] * (latest_sqrt_px - self.virtual_multiplier1)
            liq[3] = blk_num
        return last_res

class PearlV2Pool:
    X96 = 2**96
    tick_to_px_cache = {}
    tuple_to_pool = {}
    pool_to_tokenIds = {}
    tokens_metadata = {}
    nft_length = 0
    nft_manager = None

    @classmethod
    def save_classdata_to_local(cls, dt_date, path='./'):
        path += f'daily_data/pearl_pools/classdata_{dt_date}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'tick_to_px_cache': cls.tick_to_px_cache,
                'tuple_to_pool': cls.tuple_to_pool,
                'pool_to_tokenIds': cls.pool_to_tokenIds,
                'tokens_metadata': cls.tokens_metadata,
                'nft_length': cls.nft_length
            }, f)

    @classmethod
    def add_pool(cls, pool_tuple, pool_address):
        cls.tuple_to_pool[pool_tuple] = pool_address
        cls.pool_to_tokenIds[pool_address] = set()

        
    @classmethod
    def set_nft_manager(cls, nft_manager):
        cls.nft_manager = nft_manager

    @classmethod
    def get_pool_by_tokenId(cls, tokenId, blk=None):
        for pool, tokens in cls.pool_to_tokenIds.items():
            if tokenId in tokens:
                return pool
        else:
            # nonce, operator, token0, token1, fee, tickLower, tickUpper, liquidity, feeGrowthInside0LastX128, feeGrowthInside1LastX128, tokensOwed0, tokensOwed1
            try:
                pos = cls.nft_manager.functions.positions(tokenId)
                if blk is not None:
                    try:
                        _, _, _t0, _t1, _fee, tickLower, tickUpper, _, _, _, _, _ = pos.call(block_identifier=blk)
                    except Exception as e:
                        _, _, _t0, _t1, _fee, tickLower, tickUpper, _, _, _, _, _ = pos.call(block_identifier=blk-2) # in case burn
                else:
                    _, _, _t0, _t1, _fee, tickLower, tickUpper, _, _, _, _, _ = pos.call()
            except Exception as e:
                print(e)
                print(tokenId, blk)
                raise ValueError("not a vaid nft token Id")
            if (_t0, _t1, _fee) not in cls.tuple_to_pool:
                print(_t0, _t1, _fee, tokenId, cls.tuple_to_pool)
                raise ValueError("Pool not found")
            pool = cls.tuple_to_pool[(_t0, _t1, _fee)]
            cls.mint_new_nft(pool, tokenId)
            cls.tokens_metadata[tokenId] = (tickLower, tickUpper)
            return pool

    
    @classmethod
    def mint_new_nft(cls, pool, tokenId):
        cls.pool_to_tokenIds[pool].add(tokenId)
        cls.nft_length += 1

    def __init__(self, token0, token1, fee, sqrtPriceX96, tick0, pool_contract, block_num):
        self.token0 = token0
        self.token1 = token1
        self.fee = fee
        self.contract = pool_contract
        self.init_blk = block_num # for price, wtd res rely on block within range
        self.latest_tick = tick0
        self.latest_sqrtPriceX96 = sqrtPriceX96
        self.tokenId_to_key = {} # Dict[uint, Tuple[owner, gauged]]
        self.ranges_by_ticks = {}
        self.upper_ranges_by_lower = [] # sorted List[Tuple[int, int]]
        self.lower_ranges_by_upper = []
        self.in_ranges_by_lower = []
        self.in_ranges_by_upper = []
        self.box = None
    
    def isTokenIn(self, token_id):
        return token_id == self.token0 or token_id == self.token1
    
    def set_box(self, box_addr):
        self.box = box_addr

    @classmethod
    def get_tick_to_sqrtpx(cls, tick):
        if tick not in cls.tick_to_px_cache:
            cls.tick_to_px_cache[tick] = (1.0001 ** (tick/2))
        return cls.tick_to_px_cache[tick]
    

    def transfer_token(self, tokenId, new_owner, gauged):
        if tokenId in self.tokenId_to_key:
            owner, gauged = self.tokenId_to_key[tokenId]
            tickLower, tickUpper = self.tokens_metadata[tokenId]
            last_res = [owner] + self.ranges_by_ticks[(tickLower, tickUpper)].get_reserves_by_id(tokenId) + [gauged]
            if gauged:
                self.tokenId_to_key[tokenId][1] = gauged
            else:
                self.tokenId_to_key[tokenId][0] = new_owner
            return last_res
        else:
            self.tokenId_to_key[tokenId] = [new_owner, gauged]
            return None
    
    def mint_token(self, amount, block, tokenId):
        owner, gauged = self.tokenId_to_key[tokenId]
        tickLower, tickUpper = self.tokens_metadata[tokenId]
        _rng = (tickLower, tickUpper)
        if _rng not in self.ranges_by_ticks:
            self.ranges_by_ticks[_rng] = RangeInPx(self.get_tick_to_sqrtpx(tickLower), self.get_tick_to_sqrtpx(tickUpper))
        last_res = self.ranges_by_ticks[_rng].get_reserves_by_id(tokenId)
        if last_res is not None:
            last_res = [owner] + last_res + [gauged]
        if self.latest_tick < tickLower:
            self.ranges_by_ticks[_rng].change_liquidity_lower(tokenId, amount, block)
            bisect.insort(self.upper_ranges_by_lower, _rng)
        elif self.latest_tick > tickUpper:
            self.ranges_by_ticks[_rng].change_liquidity_upper(tokenId, amount, block)
            bisect.insort(self.lower_ranges_by_upper, (tickUpper, tickLower))
        else:
            self.ranges_by_ticks[_rng].change_liquidity_in_range(tokenId, amount, self.get_tick_to_sqrtpx(self.latest_tick), block)
            bisect.insort(self.in_ranges_by_lower, _rng)
            bisect.insort(self.in_ranges_by_upper, (tickUpper, tickLower))
        return last_res
    
    def burn_token(self, amount, block, tokenId):
        if tokenId in self.tokens_metadata:
            _rng = self.tokens_metadata[tokenId]
            owner, gauged = self.tokenId_to_key[tokenId]
            last_owner_res = [owner] + self.ranges_by_ticks[_rng].get_reserves_by_id(tokenId) + [gauged] 
            self.ranges_by_ticks[_rng].remove_liquidity(tokenId, amount, block)
            if self.ranges_by_ticks[_rng].is_empty():
                del self.ranges_by_ticks[_rng]
                if self.latest_tick < _rng[0]:
                    del self.upper_ranges_by_lower[bisect.bisect_left(self.upper_ranges_by_lower, _rng)]
                elif self.latest_tick > _rng[1]:
                    del self.lower_ranges_by_upper[bisect.bisect_left(self.lower_ranges_by_upper, (_rng[1], _rng[0]))]
                else:
                    del self.in_ranges_by_lower[bisect.bisect_left(self.in_ranges_by_lower, _rng)]
                    del self.in_ranges_by_upper[bisect.bisect_left(self.in_ranges_by_upper, (_rng[1], _rng[0]))]
            return last_owner_res
        else:
            print("Token not found")
            return None
    
    def get_current_reserves(self, latest_blk_num):
        res = []
        for token_id, (owner, gauged) in self.tokenId_to_key.items():
            _rng = self.tokens_metadata[token_id]
            if _rng in self.ranges_by_ticks:
                last_blk_num = self.ranges_by_ticks[_rng].owner_to_liquidity[token_id][3]
                last_amt0 = self.ranges_by_ticks[_rng].owner_to_liquidity[token_id][1]
                last_amt1 = self.ranges_by_ticks[_rng].owner_to_liquidity[token_id][2]
                res.append((owner, last_amt0, last_amt1, last_blk_num, gauged))
                self.ranges_by_ticks[_rng].owner_to_liquidity[token_id][3] = latest_blk_num
            else:
                print("Range not found when get_current_reserves!!!")
        return res

    def add_range(self, amount, block, owner=None, token_id=None, tickLower=None, tickUpper=None):
        if token_id is None:
            token_id = owner
        if tickLower is None or tickUpper is None:
            if token_id not in self.tokens_metadata:
                print("Token not found")
                return None
            tickLower, tickUpper = self.tokens_metadata[token_id]
            _rng = (tickLower, tickUpper)
        else:
            _rng = (tickLower, tickUpper)
            self.tokens_metadata[token_id] = _rng
        if _rng not in self.ranges_by_ticks:
            print('add range', _rng)
            self.ranges_by_ticks[_rng] = RangeInPx(self.get_tick_to_sqrtpx(tickLower), self.get_tick_to_sqrtpx(tickUpper))
        if self.latest_tick < tickLower:
            self.ranges_by_ticks[_rng].add_liquidity_lower(token_id, amount, block)
            self.ranges_by_ticks[_rng].set_owner(token_id, owner)
            bisect.insort(self.upper_ranges_by_lower, _rng)
        elif self.latest_tick > tickUpper:
            self.ranges_by_ticks[_rng].add_liquidity_upper(token_id, amount, block)
            self.ranges_by_ticks[_rng].set_owner(token_id, owner)
            bisect.insort(self.lower_ranges_by_upper, (tickUpper, tickLower))
        else:
            self.ranges_by_ticks[_rng].add_liquidity_in_range(token_id, amount, self.get_tick_to_sqrtpx(self.latest_tick), block)
            self.ranges_by_ticks[_rng].set_owner(token_id, owner)
            bisect.insort(self.in_ranges_by_lower, _rng)
            bisect.insort(self.in_ranges_by_upper, (tickUpper, tickLower))
    
    def remove_range(self, tickLower, tickUpper, token_id, amount, block):
        _rng = (tickLower, tickUpper)
        if _rng not in self.ranges_by_ticks:
            print("Range not found")
            return None
        last_owner_res = self.ranges_by_ticks[_rng].remove_liquidity(token_id, amount, block)
        if self.ranges_by_ticks[_rng].is_empty():
            print('remove range', _rng)
            del self.ranges_by_ticks[_rng]
            if self.latest_tick < tickLower:
                del self.upper_ranges_by_lower[bisect.bisect_left(self.upper_ranges_by_lower, _rng)]
            elif self.latest_tick > tickUpper:
                del self.lower_ranges_by_upper[bisect.bisect_left(self.lower_ranges_by_upper, (tickUpper, tickLower))]
            else:
                del self.in_ranges_by_lower[bisect.bisect_left(self.in_ranges_by_lower, _rng)]
                del self.in_ranges_by_upper[bisect.bisect_left(self.in_ranges_by_upper, (tickUpper, tickLower))]
        return last_owner_res
    
    def tick_change(self, new_tick, blk_num):
        last_res_tuples = []
        if new_tick < self.latest_tick: # tick down, more res0, less res1
            new_ix = bisect.bisect_left(self.lower_ranges_by_upper, (new_tick, new_tick)) # [new_tick, new_tick] > any range end with new_tick
            for _ut, _lt in self.lower_ranges_by_upper[new_ix:]:
                _rng = (_lt, _ut)
                if _lt >= new_tick:
                    last_res_tuples += self.ranges_by_ticks[_rng].get_res_lower_range(blk_num)
                    bisect.insort(self.upper_ranges_by_lower, _rng)
                else:
                    last_res_tuples += self.ranges_by_ticks[_rng].price_change_in_range(self.get_tick_to_sqrtpx(new_tick), blk_num)
                    bisect.insort(self.in_ranges_by_lower, _rng)
                    bisect.insort(self.in_ranges_by_upper, (_ut, _lt))
            self.lower_ranges_by_upper = self.lower_ranges_by_upper[:new_ix]
            new_ix = bisect.bisect_left(self.in_ranges_by_lower, (new_tick, new_tick))
            for _rng in self.in_ranges_by_lower[new_ix:]:
                # _rng = (_lt, _ut)
                last_res_tuples += self.ranges_by_ticks[_rng].get_res_lower_range(blk_num)
                bisect.insort(self.upper_ranges_by_lower, _rng)
                _ix_to_pop = bisect.bisect_left(self.in_ranges_by_upper, (_rng[1], _rng[0]))
                del self.in_ranges_by_upper[_ix_to_pop]
            self.in_ranges_by_lower = self.in_ranges_by_lower[:new_ix]
        elif new_tick > self.latest_tick: # tick up, less res0, more res1
            new_ix = bisect.bisect_left(self.upper_ranges_by_lower, (new_tick, new_tick))
            for _lt, _ut in self.upper_ranges_by_lower[:new_ix]:
                _rng = (_lt, _ut)
                if _ut <= new_tick:
                    last_res_tuples += self.ranges_by_ticks[_rng].get_res_upper_range(blk_num)
                    bisect.insort(self.lower_ranges_by_upper, (_ut, _lt))
                else:
                    last_res_tuples += self.ranges_by_ticks[_rng].price_change_in_range(self.get_tick_to_sqrtpx(new_tick), blk_num)
                    bisect.insort(self.in_ranges_by_upper, (_ut, _lt))
                    bisect.insort(self.in_ranges_by_lower, _rng)
            self.upper_ranges_by_lower = self.upper_ranges_by_lower[new_ix:]
            new_ix = bisect.bisect_left(self.in_ranges_by_upper, (new_tick, new_tick))
            for _ut, _lt in self.in_ranges_by_upper[:new_ix]:
                last_res_tuples += self.ranges_by_ticks[(_lt, _ut)].get_res_upper_range(blk_num)
                bisect.insort(self.lower_ranges_by_upper, (_ut, _lt))
                _ix_to_pop = bisect.bisect_left(self.in_ranges_by_lower, (_lt, _ut))
                del self.in_ranges_by_lower[_ix_to_pop]
            self.in_ranges_by_upper = self.in_ranges_by_upper[new_ix:]
        self.latest_tick = new_tick
        res = []
        for tid, amt0, amt1, blk_num in last_res_tuples:
            owner, gauged = self.tokenId_to_key[tid]
            res.append([owner, amt0, amt1, blk_num, gauged])
        return res


class AlmPositions:

    def __init__(self, box_address, gauge_address):
        self.box_address = box_address
        self.gauge_address = gauge_address
        self.tick_lower = None
        self.tick_upper = None
        self.liquidity = 0
        self.balance0_left = 0
        self.balance1_left = 0
        self.fees_to_vault0 = 0
        self.fees_to_vault1 = 0
        self.total_supply = 0
        self.latest_tick = None
        self.owner_to_shares = dict()
        self.last_blk = -1

        self.lower_sqrt_px = 1
        self.upper_sqrt_px = 1
        self.virtual_multiplier0 = 0
        self.virtual_multiplier1 = 0
        self.max_multiplier0 = 0
        self.max_multiplier1 = 0    
    
    def _get_latest_dp(self):
        if self.latest_tick is None or self.tick_lower is None or self.tick_upper is None:
            return None
        latest_sqrt_px = PearlV2Pool.get_tick_to_sqrtpx(self.latest_tick)
        if self.latest_tick < self.tick_lower:
            dp0 = self.max_multiplier0
            dp1 = 0
        elif self.latest_tick > self.tick_upper:
            dp0 = 0
            dp1 = self.max_multiplier1
        else:
            dp0 = (1 / latest_sqrt_px - self.virtual_multiplier0)
            dp1 = (latest_sqrt_px - self.virtual_multiplier1)
        return dp0, dp1
    
    def get_current_reserves(self, blk_num, new_tick=None):
        res = []
        # print('get_current_reserves', self.box_address, blk_num, self.last_blk, self.total_supply)
        if self.last_blk < blk_num and self.total_supply > 0:
            dps = self._get_latest_dp()
            if dps is not None:
                for owner, shs in self.owner_to_shares.items():
                    bal0 = (dps[0] * self.liquidity + self.balance0_left - self.fees_to_vault0) / self.total_supply * shs[0]
                    bal1 = (dps[1] * self.liquidity + self.balance1_left - self.fees_to_vault1) / self.total_supply * shs[0]
                    if bal0 < 0:
                        print('negative bal0 in box',self.box_address, dps[0], self.liquidity, self.balance0_left, self.fees_to_vault0, self.total_supply, shs[0])
                    if bal1 < 0:
                        print('negative bal1 in box',self.box_address, dps[0], self.liquidity, self.balance1_left, self.fees_to_vault1, self.total_supply, shs[0])
                    res.append([owner, bal0, bal1, shs[1]])
                    shs[1] = blk_num
            else:
                for owner, shs in self.owner_to_shares.items():
                    bal0 = (self.balance0_left - self.fees_to_vault0) / self.total_supply * shs[0]
                    bal1 = (self.balance1_left - self.fees_to_vault1) / self.total_supply * shs[0]
                    if bal0 < 0:
                        print('negative bal0 in box', self.box_address, self.balance0_left, self.fees_to_vault0, self.total_supply, shs[0])
                    if bal1 < 0:
                        print('negative bal1 in box', self.box_address, self.balance1_left, self.fees_to_vault1, self.total_supply, shs[0])
                    res.append([owner, bal0, bal1, shs[1]])
                    shs[1] = blk_num
        self.last_blk = blk_num
        if new_tick is not None:
            self.latest_tick = new_tick
        return res

    def pool_price_change(self, new_tick, blk_num):
        # todo:
        res = [blk_num, 0, 0, 0]
        if new_tick > self.tick_upper:
            res[1] = self.max_multiplier0
        pass
    
    def pool_change(self, tick_lower, tick_upper, liquidity, blk_num):
        res = self.get_current_reserves(blk_num)
        self.tick_lower = tick_lower
        self.tick_upper = tick_upper
        self.liquidity += liquidity
        self.liquidity = max(0, self.liquidity)
        self.lower_sqrt_px = PearlV2Pool.get_tick_to_sqrtpx(tick_lower)
        self.upper_sqrt_px = PearlV2Pool.get_tick_to_sqrtpx(tick_upper)
        self.virtual_multiplier0 = 1 / self.upper_sqrt_px
        self.virtual_multiplier1 = self.lower_sqrt_px
        self.max_multiplier0 = 1 / self.lower_sqrt_px - self.virtual_multiplier0
        self.max_multiplier1 = self.upper_sqrt_px - self.virtual_multiplier1
        return res
    
    def box_token_transfer(self, from_addr, to_addr, shares, blk_num):
        dps = self._get_latest_dp()
        res = []
        if from_addr == self.gauge_address or to_addr == self.gauge_address:
            return res
        if from_addr != '':
            if from_addr not in self.owner_to_shares:
                print("Owner not found")
                raise
            shs = self.owner_to_shares[from_addr]
            if dps is not None and self.total_supply > 0:
                bal0 = (dps[0] * self.liquidity + self.balance0_left - self.fees_to_vault0) / self.total_supply * shs[0]
                bal1 = (dps[1] * self.liquidity + self.balance1_left - self.fees_to_vault1) / self.total_supply * shs[0]
                res.append([from_addr, bal0, bal1, shs[1]])
            shs[0] -= shares
            if shs[0] <= 0:
                del self.owner_to_shares[from_addr]
            else:
                shs[1] = blk_num
        else:
            self.total_supply += shares
        if to_addr != '':
            if to_addr not in self.owner_to_shares:
                self.owner_to_shares[to_addr] = [shares, blk_num]
            else:
                shs = self.owner_to_shares[to_addr]
                if dps is not None and self.total_supply > 0:
                    bal0 = (dps[0] * self.liquidity + self.balance0_left - self.fees_to_vault0) / self.total_supply * shs[0]
                    bal1 = (dps[1] * self.liquidity + self.balance1_left - self.fees_to_vault1) / self.total_supply * shs[0]
                    res.append([to_addr, bal0, bal1, shs[1]])
                shs[0] += shares
                shs[1] = blk_num
        else:
            self.total_supply -= shares
        return res
    
    def box_balance0_change(self, delta_balance0, blk_num):
        res = self.get_current_reserves(blk_num)
        self.balance0_left += delta_balance0
        self.balance0_left = max(0, self.balance0_left)
        return res
    
    def box_balance1_change(self, delta_balance1, blk_num):
        res = self.get_current_reserves(blk_num)
        self.balance1_left += delta_balance1
        self.balance1_left = max(0, self.balance1_left)
        return res

    def set_box_fees_to_vault(self, fees_to_vault0, fees_to_vault1, blk_num):
        res = self.get_current_reserves(blk_num)
        return res