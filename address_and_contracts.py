import json
from web3 import Web3


RWA_PATH = './'
REAL_RPC_ENDPOINT = 'https://real.drpc.org'
W3 = Web3(Web3.HTTPProvider(REAL_RPC_ENDPOINT))

def test_rpc_connection():
    print(f'connected: {W3.is_connected()}, chain_id: {W3.eth.chain_id}, highest_block_number: {W3.eth.block_number}')

NULL_ADDR = '0x0000000000000000000000000000000000000000'
REETH_ADDRESS = '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE'
WREETH_ADDRESS = '0x90c6E93849E06EC7478ba24522329d14A5954Df4'
USTB_ADDRESS = '0x83feDBc0B85c6e29B589aA6BdefB1Cc581935ECD'
CVR_ADDRESS = '0xB08F026f8a096E6d92eb5BcbE102c273A7a2d51C'
PEARL_ADDRESS = '0xCE1581d7b4bA40176f0e219b2CaC30088Ad50C7A'
VEPEARL_ADDRESS = '0x99E35808207986593531D3D54D898978dB4E5B04'
VEPEARL_VESTING_ADDRESS = '0xC06b2cE291aaA69C6FC0be8a2B33868aAF7a1950'
RWA_ADDRESS = '0x4644066f535Ead0cde82D209dF78d94572fCbf14'
VERWA_ADDRESS = '0xa7B4E29BdFf073641991b44B283FD77be9D7c0F4'
VERWA_VESTING_ADDRESS = '0x2298963e41849a46A5b4aD199231edDD9ecce958', # VE Vesting RWA
MORE_ADDRESS = '0x25ea98ac87A38142561eA70143fd44c4772A16b6'
DAI_ADDRESS = '0x75d0cBF342060b14c2fC756fd6E717dFeb5B1B70'
ARCUSD_ADDRESS = '0xAEC9e50e3397f9ddC635C6c429C8C7eca418a143'
PTA_ADDRESS = '0xeAcFaA73D34343FcD57a1B3eB5B0D949df727712'

PEARLV2_FACTORY_ADDRESS = '0xeF0b0a33815146b599A8D4d3215B18447F2A8101'

OLD_LIQUID_BOX_FACTORY_ADDRESS = '0xD25dDDb75cAD1f31138EfbE5300142B661A37272'
LIQUID_BOX_FACTORY_ADDRESS = '0xEfeFFa36b9FA787b500b2d18a3829271526023b1'
OLD_LIQUID_BOX_MANAGER_ADDRESS = '0x17fA0AC6B2781ac85540C35Bb6d3610245EF7e92'
LIQUID_BOX_MANAGER_ADDRESS = '0x644FA21b6aDc1f73E94eA2984E116fC065394E33'
VOTER_ADDRESS = '0x4C44cFBBc35e68a31b693b3926F11a40abA5f93B'
PEARL_NFT_MANAGER_ADDRESS = "0x153e99930da597EA11144327afC6Ae5E6f853575" # "0x6684c6D8AaB4784FD31db3fdd28Ed0d098C3Db91"
USDC_ADDRESS = '0x8D7dd0C2FbfAF1007C733882Cd0ccAdEFFf275D2'
USDT_ADDRESS = '0xDDF533a1Cd8376473Bfe5ae1d93b90e39e3D6faD'
WBTC_ADDRESS = '0x4deE73429D25E92E9c7e7e580e914820C3Abdc8D'
CVR_STAKING_CHEF_ADDRESS = '0x2Ecd92e75A611A92233F434253364DDaC95Eb55A'
PEARL_SWAP_ROUTER_ADDRESS = '0xa1F56f72b0320179b01A947A5F78678E8F96F8EC'

PEARLV2_POOL_ADDRESS = '0xd27F8A40222c30F3359b83e8Ab5Ce5295d441A0A'

PEARL_QUOTERV2_ADDRESS = '0xDe43aBe37aB3b5202c22422795A527151d65Eb18'
SMORE_ADDRESS = '0xD1e39288520f9f3619714B525e1fD5F8c023dbA1'

VAULT_FACTORY_ADDRESS = '0x303C5d72D2d123ac6C36957d167Ca7Cfee3414e7'

UKRE_ADDRESS = '0x835d3E1C0aA079C6164AAd21DCb23E60eb71AF48'
BRIBE_FACTORY = '0x4fc90a1Eb14988a6735c9Aedb5Bd38915EcCaaC4'

ROUTE_TO_USTB = {
    PEARL_ADDRESS: ['1x35BA384F9D30D68028898849ddBf5bda09bbE7EA'],
    RWA_ADDRESS: ['0x182d3F8e154EB43d5f361a39A2234A84508244c9','1x5dfa942B42841Dd18883838D8F4e5f7d8CEb5Eeb'],
    USTB_ADDRESS: [],
    ARCUSD_ADDRESS: ['1xC6B3AaaAbf2f6eD6cF7fdFFfb0DaC45E10c4A5B3'],
    WREETH_ADDRESS: ['1x5dfa942B42841Dd18883838D8F4e5f7d8CEb5Eeb'],
    DAI_ADDRESS: ['0x727B8b6135dcFe1E18A2689aBBe776a6810E763c'],
    USDC_ADDRESS: ['1xC768956E9E03950Ee10a37C064ba91da38235Ecc'],
    MORE_ADDRESS: ['0x6b1a34df762f1d3367e7e93AE5661c88CA848423'],
    CVR_ADDRESS: ['0xfA88A4a7fF6D776c3D0A637095d7a9a4ed813872', '1x35BA384F9D30D68028898849ddBf5bda09bbE7EA'],
    UKRE_ADDRESS: ['0x72c20EBBffaE1fe4E9C759b326D97763E218F9F6', '1xC6B3AaaAbf2f6eD6cF7fdFFfb0DaC45E10c4A5B3']
}

abis = {}
abis['basket_token'] = json.load(open(RWA_PATH + 'abi/basket_token.json', 'rb'))
# abis['erc20'] = json.load(open('./abi/erc20.json', 'rb'))
abis['erc20'] = json.load(open(RWA_PATH + 'abi/erc20.json', 'rb'))
abis['erc1967'] = json.load(open(RWA_PATH + 'abi/erc1967.json', 'rb'))
abis['pearl_factory'] = json.load(open(RWA_PATH + 'abi/pearl_factory.json', 'rb'))
abis['pearlv2_pool'] = json.load(open(RWA_PATH + 'abi/pearlv2_pool.json', 'rb'))
abis['pearl_nft_manager'] = json.load(open(RWA_PATH + 'abi/pearl_nft_manager.json', 'rb'))
abis['gauge'] = json.load(open(RWA_PATH + 'abi/gauge.json', 'rb'))
abis['voter'] = json.load(open(RWA_PATH + 'abi/voter.json', 'rb'))
abis['vault_factory'] = json.load(open(RWA_PATH + 'abi/vault_factory.json', 'rb'))
abis['stack_vault'] = json.load(open(RWA_PATH + 'abi/stack_vault.json', 'rb'))
abis['smore'] = json.load(open(RWA_PATH + 'abi/smore.json', 'rb'))
abis['cvr_staking_chef'] = json.load(open(RWA_PATH + 'abi/cvr_staking_chef.json', 'rb'))
abis['liquid_box_manager'] = json.load(open(RWA_PATH + 'abi/liquid_box_manager.json', 'rb'))
abis['liquid_box_factory'] = json.load(open(RWA_PATH + 'abi/liquid_box_factory.json', 'rb'))
abis['liquid_box'] = json.load(open(RWA_PATH + 'abi/liquid_box.json', 'rb'))
abis['pta'] = json.load(open(RWA_PATH + 'abi/arcUSD_point_vault.json', 'rb'))

address_to_symbol = {}
contracts= {}
contracts['rwa'] = W3.eth.contract(address=RWA_ADDRESS, abi=abis['erc20'])
address_to_symbol[RWA_ADDRESS] = 'rwa'
contracts['verwa'] = W3.eth.contract(address=VERWA_ADDRESS, abi=abis['erc1967'])
address_to_symbol[VERWA_ADDRESS] = 'verwa'
contracts['pearl'] = W3.eth.contract(address=PEARL_ADDRESS, abi=abis['erc20'])
address_to_symbol[PEARL_ADDRESS] = 'pearl'
contracts['vepearl'] = W3.eth.contract(address=VEPEARL_ADDRESS, abi=abis['erc1967'])
address_to_symbol[VEPEARL_ADDRESS] = 'vepearl'
contracts['wreeth'] = W3.eth.contract(address=WREETH_ADDRESS, abi=abis['erc20'])
address_to_symbol[WREETH_ADDRESS] = 'wreeth'
address_to_symbol[REETH_ADDRESS] = 'reeth'
# contracts['wbtc'] = W3.eth.contract(address=WBTC_ADDRESS, abi=abis['erc20'])
# address_to_symbol[WBTC_ADDRESS] = 'wbtc'
contracts['ustb'] = W3.eth.contract(address=USTB_ADDRESS, abi=abis['erc20'])
address_to_symbol[USTB_ADDRESS] = 'ustb'
contracts['ukre'] = W3.eth.contract(address=UKRE_ADDRESS, abi=abis['erc20'])
address_to_symbol[UKRE_ADDRESS] = 'ukre'
contracts['arcusd'] = W3.eth.contract(address=ARCUSD_ADDRESS, abi=abis['erc20'])
address_to_symbol[ARCUSD_ADDRESS] = 'arcusd'
contracts['pta'] = W3.eth.contract(address=PTA_ADDRESS, abi=abis['pta'])
address_to_symbol[PTA_ADDRESS] = 'pta'
contracts['dai'] = W3.eth.contract(address=DAI_ADDRESS, abi=abis['erc20'])
address_to_symbol[DAI_ADDRESS] = 'dai'
contracts['more'] = W3.eth.contract(address=MORE_ADDRESS, abi=abis['erc20'])
address_to_symbol[MORE_ADDRESS] = 'more'
contracts['vault_factory'] = W3.eth.contract(address=VAULT_FACTORY_ADDRESS, abi=abis['vault_factory'])
contracts['smore'] = W3.eth.contract(address=SMORE_ADDRESS, abi=abis['smore'])
contracts['cvr'] = W3.eth.contract(address=CVR_ADDRESS, abi=abis['erc20'])
address_to_symbol[CVR_ADDRESS] = 'cvr'
contracts['cvr_staking_chef'] = W3.eth.contract(address=CVR_STAKING_CHEF_ADDRESS, abi=abis['cvr_staking_chef'])
contracts['pearl_factory'] = W3.eth.contract(address=PEARLV2_FACTORY_ADDRESS, abi=abis['pearl_factory'])
contracts['pearl_nft_manager'] = W3.eth.contract(address=PEARL_NFT_MANAGER_ADDRESS, abi=abis['pearl_nft_manager'])
contracts['voter'] = W3.eth.contract(address=VOTER_ADDRESS, abi=abis['voter'])
contracts['liquid_box_manager'] = W3.eth.contract(address=LIQUID_BOX_MANAGER_ADDRESS, abi=abis['liquid_box_manager'])
contracts['liquid_box_factory'] = W3.eth.contract(address=LIQUID_BOX_FACTORY_ADDRESS, abi=abis['liquid_box_factory'])
# contracts['bribe_factory'] = W3.eth.contract(address=BRIBE_FACTORY, abi=abis['bribe_factory'])

BLACKLIST = set([
    '',
    NULL_ADDR,
    '0x000000000000000000000000000000000000dEaD',
    '0x90c6E93849E06EC7478ba24522329d14A5954Df4', # WREETH
    '0x048C661327F13dB139CBD5C231e62810587cE1F0', # VOTER_ADDRESS
    '0x2995ab3D8219Cf6e240850422BCda19b3A2eaA2D', # Minter
    '0x182d3F8e154EB43d5f361a39A2234A84508244c9', # pool rwa wreeth
    '0x35BA384F9D30D68028898849ddBf5bda09bbE7EA', # pool pearl ustb
    '0x5dfa942B42841Dd18883838D8F4e5f7d8CEb5Eeb', # pool ustb wreeth
    '0x6b1a34df762f1d3367e7e93AE5661c88CA848423', # pool ustb more
    '0x727B8b6135dcFe1E18A2689aBBe776a6810E763c', # pool dai ustb
    '0xC6B3AaaAbf2f6eD6cF7fdFFfb0DaC45E10c4A5B3', # pool arcusd ustb0x99E35808207986593531D3D54D898978dB4E5B04
    '0xC768956E9E03950Ee10a37C064ba91da38235Ecc', # pool ustb usdc
    '0xEc38388d79a90Fe4AE49cE9a984d299C9Cf8a36F', # pool dai ustb
    '0xfA88A4a7fF6D776c3D0A637095d7a9a4ed813872', # pool cvr-pearl
    '0x72c20EBBffaE1fe4E9C759b326D97763E218F9F6', # pearl pool arcusd ukre
    '0x9C55013DDb753287Be17A109A4EcC80d0d8a3Ae0', # GaugeV2 arusd ustb
    '0xCCF94837618CF7046d913f6A5207Ef24d6d52D45', # GaugeV2 more ustb
    '0x788133311cB08329794B81076A3c9d22FED66222', # GaugeV2 rwa wreeth
    '0xBEE6F8087afF9f92b7EeeA2bf50FB3B305890C1E', # GaugeV2 ustb wreeth
    '0xBDDf572B4B61E705466C94EcE60d1e0b8B42Ab20', # GaugeV2 dai ustb
    '0x7fbb6AE37f74e2f6D73618Ff45014f3F095D179c', # Gauge pearl ustb
    '0xED331358284733f2893f0e14cC298178E59fB468', # Gauge Pearl cvr
    '0xC26ebA08331b5732f75a6FE479aaDA43a3195945', # GaugeV2ALM 
    '0x5Db8EFFaCA3A63e7450633BdEb27300878356896', # GaugeV2ALM
    '0xd86288dA2aE1C4Bf3597bd213453ace1c92b6888', # GaugeV2ALM Trident USTB-arcUSD
    '0xaD325c75922338caBc33550Df4B9eA996E27c946', # GaugeV2ALM Trident RWA_WREETH
    '0x8e30cDB64E018dAa0325fb6d7635cFaEc5B53E29', # GaugeV2ALM  trident ustb-pearl
    '0x6815af1945320e954EE689cfb07fd30fb581D65b', # GaugeV2ALM
    '0xbd8fd95C6ec5138fa00e2DcACC5C3d954aC2429c', # GaugeV2ALM
    '0x089CabE665C5e45b49Bf5B50d1C00A510DB8BD26', # Trident RWA-WREETH
    '0x39489A11E45D468cFd434E653b0EFD1721Ea077a', # Trident USTB-PEARL
    '0x680e46014162dF835b70424466523089dbfeA1D9', # Trident USTB-WREETH
    '0x6196bdFBCa61913972562e7C19f390614B90a492', # Trident DAI-USTB
    '0xe0F09F092ae3ba914f1659213FACb97309805Ca9', # Trident USTB-arcUSD
    '0x2485bA1E2A46e6E997794514BE0caBe20d749979', # trident dai ustb
    '0x1A86E3c934BFb2Df3b2e3734fBf7e620Ce5B1878', # trident ustb-arcusd
    '0xA7DF3FCaA03d21b35f18Ac13B2c1164BFd1472B8', # Trident UKRE-arcusd
    '0xA7dFf276F939B9129A92B4262D952964c4BDf95F', # Trident RWA-WREETH
    '0xa77cb64Ee2ecF17D735a3b1b9820131E41758b50', # Trident USTB-PEARL
    '0xdd3324B644348C907146C0A7527a32384F2959fd', # trident cvr-pearl
    '0xe9D7686bc30e867CEB953c24e5e33a0f77A87697', # trident ustb-wreeth
    '0x76fd8777b0E34fAfeC162F01bC183566b2cf1EFa', # trident more ustb
    '0xA56ec33bBda9A67269D303Be1d6a06e36785928c', # Bribe rwa wreeth
    '0xD5D5731f48aB62d8C5C61b16eeD2b744d0e6d260', # Bribe Pearl ustb
    '0xa71E1B2d55267f734F28f1679eA78aEF1069E376', # Bribe USTB wreeth
    '0xCE92127078fC6A34d977B05989901bf66c395A51', # Bribe
    '0x86a08F4BCd5677CBDe543868aAE288640B6e12Ff', # Bribe ustb dai
    '0x5018ECbAD4a7aa5e77AbF0B76a9835d6C454A37D', # Bribe
    '0xBe812F9Df6de9b0C1c430f23942455F5626afFE5', # Bribe
    '0xC2eea5774078727f5EFAB59F3960138D9d687592', # Bribe ustb dai
    '0x00E87216Cf49893f7aDD2290d91A1429237b8F3d', # bribe ustb dai
    '0x0c55B4b65D37Bf43396827c3855D82BF9e76c5Fc', # Bribe USTB
    '0x28dF5a8fbbFa6a00542CC8eF4aF985E987cC760D', # Bribe USTB DAI
    '0x2e8a097F4257cc1C7ed43402f6958e7735A92124', # Bribe USTB DAI
    '0x42B9ed82Df1b25a382134D117A172445D0196439', # Bribe USTB
    '0x6Fb330766E1F6Ef0cB93b83C14B990b43E9b32fD', # Bribe USTB DAI
    '0xFd07c3FdD582477a907F9977F9D24eB76f9Dc6e1', # Bribe CVR PEARL
    '0xa866dA53E5420af43BB92b0324B3BBDD89d0ff3e', # Bribe
    '0xf2b9E8c96c813130D3937A7c0aF08c61C5054E7F', # Bribe
    '0x1E6844C681B1b1A645a70673401616AE221aF1F4', # Bribe
    '0xB7624A3F54e65A5EB7400BBdc88067a08A042Fe9', # Bribe ustb wreeth
    '0x549cf22DBDa4c3Fd1Cf051eF6B9fec96E37F4850', # Bribe ustb
    '0x02Ee71Ca574D62d72e201b0Da6332cEDAF252362', # Bribe USTB CVR
    '0x0688Db64637ed352a4dcf88600b317bf389bBCE6', # Bribe USTB DAI
    '0x0d223008469aA7d699Ac4EF98a4eB9694C6071C6', # Bribe USTB
    '0x1Cd084C5CC9525DEa8b50C59096EdFAfC412616A', # Bribe RWA WREETH
    '0x1F40AB61Db53B267c057601f46b965a4fC16bcEB', # Bribe USTB
    '0x27327C0c64Bf889D94f0cC546b8e58603534fb4e', # Bribe ustb arcusd
    '0x375F460e4EBAA7564d9630cB729E54E3fb16B0b3', # Bribe USTB DAI
    '0x789CdBDd84136bdD88Cf0eE672b750ADef762AFD', # Bribe ustb
    '0x87b9772FEedF421437d15d9CAac3c1f9Eb984a04', # Bribe ustb more
    '0x90a7C42534bE312e45Cf2db40a2d6B52BD65eA82', # Bribe ustb arcusd
    '0x3d6EB192EEEdD6bEc1729a54fCaD2C5F6C72Ef06', # CaviarManager
    '0x2fB0CA16B289e979F908927Cb4DBab5923C52111', # CavierStrategy
    '0x04348139c1EC82beF38163463ce0389B0b7140a7', # CaviarFeeManager
    '0x2Ecd92e75A611A92233F434253364DDaC95Eb55A', # CaviarStakingChef
    '0xC68E2aFC7DF4e378aD23055526A19ba037e36B1d', # CaviarRebaseChef
    '0x303C5d72D2d123ac6C36957d167Ca7Cfee3414e7', # VAULT_FACTORY_ADDRESS
    '0x561F2826A9d2A653fdC903A9effa23c0C0c3B549', # StackVault USTB
    '0xA3949263535D40d470132Ab6CA76b16D6183FD31', # StackVault arcUSD
    '0xa5C30E10B3a769Ad6FEe5c1603235cb0C688dDcf', # StackVault pta
    '0x4928E0690F7B39b35bB0a494058492af8774c3D5', # StackVault reeth
    '0xb311D3999ec9B77971d3Db6ef043E7bD54CE5218', # MORE Minter
    '0xD1e39288520f9f3619714B525e1fD5F8c023dbA1', # SMORE
    '0x7a2E4F574C0c28D6641fE78197f1b460ce5E4f6C', # RevenueDistributor
    '0x9D146A1C099adEE2444aFD629c04B4cbb5eE1539', # RevenueStream rwa
    '0x27b7d3bdA2F6D656C8B37Bba7B1757eE75E6525e', # RewardsDistributor
    '0x98e73e553bd3282726890710F80279462AaB8C36', # GnosisSafeProxy
    '0x499D011d7F13c707EebEe5B677A772d853723C0F', # GnosisSafeProxy
    '0x946C569791De3283f33372731d77555083c329da', # GnosisSafeProxy
    '0x7D46545C3d82607371C22858F56E56074d395198', # GnosisSafeProxy
    '0xAC0926290232D07eD8b083F6BE3Ab040010f757F', # GnosisSafeProxy
    '0x5111e9bCb01de69aDd95FD31B0f05df51dF946F4', # GnosisSafeProxy
    '0x636e2C638fDd3593329E34E5291ecb04B0938C1B', # GnosisSafeProxy
    '0x55dBd594F19f7bE69eaC7910bD4E782D1F417820', # GnosisSafeProxy
    '0x164f1622207AaAbCd32839ED4B0d97c3387C2492', # RoyaltyHandler    
    '0x17fA0AC6B2781ac85540C35Bb6d3610245EF7e92', # OLD_LIQUID_BOX_MANAGER_ADDRESS,    
    '0x644FA21b6aDc1f73E94eA2984E116fC065394E33', # LIQUID_BOX_MANAGER_ADDRESS,    
    '0x332bCE4ff5F0C0b30205D468B44AabBFBaf3C58e', # NFTCollector    
    '0x6C2c653BCEB606bE8E7e92D008c62D0e05a83fd9', # arcUSDMinter
    '0xAEC9e50e3397f9ddC635C6c429C8C7eca418a143', # ARCUSD
    '0xD68dde232491138171cEf2d313c7404780E06455', # arcUSDFeeCollector
    '0xeAcFaA73D34343FcD57a1B3eB5B0D949df727712', # PTa
    '0xD0b3DfCB4383b10d964A4E0cb1a0Cea19C9F89AC', # Custodian Manager
    '0xa1F56f72b0320179b01A947A5F78678E8F96F8EC', # SwapRoute
    '0x835d3E1C0aA079C6164AAd21DCb23E60eb71AF48', # basket UKRE
    '0xCE1581d7b4bA40176f0e219b2CaC30088Ad50C7A', # Pearl
    '0xa7B4E29BdFf073641991b44B283FD77be9D7c0F4', # veRWA
    '0x2298963e41849a46A5b4aD199231edDD9ecce958', # VERWA_VESTING_ADDRESS
    '0x99E35808207986593531D3D54D898978dB4E5B04', # VEPEARL
    '0xC06b2cE291aaA69C6FC0be8a2B33868aAF7a1950', # VEPEARL_VESTING_ADDRESS
    '0x7DC43C0165Bfc9d202Fa24beF10992F599014999', # VeDistributor
    '0xce3D0d6A238928eB31E52478f68B4051236A3F13', # AMO
    '0x02473349D9e2AbbFcF5b82F171b55Cd694f9Fc7A', # FeeSplitter
    '0x153e99930da597EA11144327afC6Ae5E6f853575', # PEARL_NFT_MANAGER_ADDRESS,
    '0x60a6c99d0005b89c6F0E736E212004000f330aed', # Pearl Router
    '0xc6Bbd6A41F673d7AEC0E2906F14Df14D06A3b750', # ERC4626Router
    '0xbEC49aAB96e4837EAf38C55a247ac1bAfB7aF019', # GaugeV2CL, trident ustb arcusd
    '0xc2736Fa436029edEaf228ECed4bEEEf58C4d7F13', # GaugeV2CL, trident ustb more
    '0xdDF78c3C688883aE328D2522C8cf13bFC7d63745', # GaugeV2CL, trident ustb dai
    '0xecd9282708769903203bdE6cF3DC1306F063514A', # GaugeV2CL, trident rwa wreeth
    '0xAd867544b8dF195d3BAe0cD076Adb54decCED2f9', # GaugeV2CL, trident ustb wreeth
    '0x20254A0a1bd2AeA0E4beBaAebca31BB9CFc5932A', # GaugeV2CL, trident CVR pearl
    '0x3C485daDcB645fD30047848b94B8eBEA5f8BD843', # GaugeV2CL, trident ustb pearl
    '0xACFA44e1a687738Faf7182fc8ba40fEC02720348', # GaugeV2CL, trident ukre arcusd
    '0xFe48C99fD8A7cbb8d3a5257E1dCcC69e9a991A48', # Opsproxy more
    '0x4C44cFBBc35e68a31b693b3926F11a40abA5f93B', # Voter
    '0xDbDa59243973e9147f7b064921a238bbdEc8f4D3', # Rent Manager
    '0xcDA705F6EE1d3130f088f58D35fE4aC0C016059C', # Swap Router
    '0x3C888C84511f4C0a4F3Ea5eD1a16ad7F6514077e', # Uni3Swpper
    '0x00678b3b8E6525dAeE0D429BCA4CC1F16f9a071C', # IFO Harvester TDT-MORE-USTB
    '0xA1a9dB809ea4E50C634f7C39EA19AB2f6FdF7664', # Harvester TDT-RWA-WREETH  
    '0xB8243A0Fa4030B2094F0f54693fF7098d269ef50', # IFO Harvester TDT-UKRE-arcUSD
    '0xE064fB4eB8183DF6fe975cADd2e8aC9454A78BcC', # IFO Harvester TDT-CVR-PEARL
    '0xF07A7285BD3feaE537C4C46B7F26068e237CA049', # IFO Harvester TDT-USTB-WREETH
    '0xe61c27cE08834d54dF438A3C92a65e834a24CfDA', # Compounder CVR
    '0xea70db45549C54Effe11EeCAc20EB312a410922F', # PerfFeeTreasury
    '0x56743A08f09a39FBA40eE48f1C92974d849FC1e5', # AMO
    '0xC3976b35Cc366881F70e0835C1221101DBa85947', # AMO
    '0x26f66A5c235510c5977df70Ce88925b3Bc438654', # unknown
    '0x4B385B194eFfbDc8Ce4D7655d5A8737C26d75408', # Unknown
    '0x7214dd01F41bC6C536Ef47fC62580A0400FC1185', # Unknown
    '0x7b86964Ee163B0Bd205868d28542d2D720A690CC', # Unknown
    '0xD00e0D6C8DB148124becb2fc271d7f787AF6D494', # unknown
    '0x04036885a8a1DFe21E2572B619FdBe75C5Bb6d56', # unknown
    '0x1FB57aF994a03c49f9B1b7Eef938519463CdF996', # unknown
])
