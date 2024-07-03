example: 
```console
    python airdrop.py 2024-05-16 2024-06-25
```

Daily reports saved in ./daily_reports/rwa/daily_summary_{date}.csv

Tracking Tokens:
    reETH/wreETH, USTB, CVR, sCVR, PEARL, RWA, MORE, DAI, sMORE, arcUSD, PTa, UKRE
  
Tracking Contract:
  - Pearl Pools:
    - Swap: accumulating fee for FeePoints; block by block price to converting value into USTB
  - Liquid Box: token balance and fee in box position and related gauge
  - Pearl NFT Manager: token balances in nft and gauge
  - CVR staking chef: block by block assets per share
  - More Vaults: Collateral deposited
  - veRWA and vePearl: locked token amount
  
