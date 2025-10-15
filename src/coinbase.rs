use bitcoin::{
    blockdata::{
        block::{Block as BtcBlock, BlockHeader},
        script::Builder as ScriptBuilder,
        transaction::{Transaction, TxIn, TxOut},
    },
    consensus::{deserialize, serialize},
    hash_types::TxMerkleNode,
    hashes::{sha256d, Hash},
    Address, OutPoint,
};
use serde_json::Value;
use hex;
use std::str::FromStr;
use crate::merkle_root;

// ----------------- Coinbase Helpers -----------------
pub fn build_coinbase_from_template(template: &Value) -> Transaction {
    let height = template["result"]["height"].as_u64().unwrap() as u32;
    let coinbase_value = template["result"]["coinbasevalue"].as_u64().unwrap();
    let payout_addr = Address::from_str("bc1qyux0tnvrusd9deq89h8er0ml4clhdwetp6ljp2").unwrap();
    let output = TxOut {
        value: coinbase_value,
        script_pubkey: payout_addr.script_pubkey(),
    };
    let input = TxIn {
        previous_output: OutPoint::default(),
        script_sig: ScriptBuilder::new().push_int(height as i64).into_script(), // ✅ fixed
        sequence: 0xFFFFFFFF,
        witness: vec![vec![0u8; 32]],
    };
    Transaction {
        version: 2,
        lock_time: 0,
        input: vec![input],
        output: vec![output],
    }
}

pub fn insert_nonce_into_coinbase(coinbase: &mut Transaction, nonce: u32) {
    coinbase.input[0].witness[0] = nonce.to_le_bytes().to_vec();
}

pub fn assemble_block_hex(template: &Value, coinbase: &Transaction, nonce: u32) -> String {
    // Build transaction list
    let mut txs = vec![coinbase.clone()];
    if let Some(txs_json) = template["result"]["transactions"].as_array() {
        for tx in txs_json {
            let raw_tx = hex::decode(tx["data"].as_str().unwrap()).unwrap();
            txs.push(deserialize(&raw_tx).unwrap());
        }
    }

    let version = template["result"]["version"].as_u64().unwrap() as i32;
    let prevhash =
        sha256d::Hash::from_str(template["result"]["previousblockhash"].as_str().unwrap()).unwrap();
    let time = template["result"]["curtime"].as_u64().unwrap() as u32;
    let bits = u32::from_str_radix(template["result"]["bits"].as_str().unwrap(), 16).unwrap();

    // Compute Merkle root
    let nodes: Vec<TxMerkleNode> = txs
        .iter()
        .map(|tx| TxMerkleNode::from(sha256d::Hash::from_inner(tx.txid().into_inner())))
        .collect();
    let merkle_root_node: TxMerkleNode = merkle_root(nodes).into();

    // Build Bitcoin block using renamed BtcBlock
    let block = BtcBlock {
        header: BlockHeader {
            version,
            prev_blockhash: prevhash.into(),
            merkle_root: merkle_root_node,
            time,
            bits,
            nonce,
        },
        txdata: txs,
    };

    hex::encode(serialize(&block))
}

// Safe insert of GPU extra nonce with VIN guard
pub fn patch_insert_nonce_into_coinbase(coinbase: &mut serde_json::Value, nonce: u32) {
    if let Some(tx) = coinbase["result"]["transactions"].get_mut(0) {
        if let Some(vin) = tx["vin"].get_mut(0) {
            let mut script_bytes = hex::decode(vin["coinbase"].as_str().unwrap_or("")).unwrap_or_default();
            let extra_nonce = nonce.to_le_bytes().to_vec();
            let len = (1 + (nonce % 4) as usize).min(extra_nonce.len());
            script_bytes.extend_from_slice(&extra_nonce[..len]);
            if script_bytes.len() < 32 {
                script_bytes.extend(vec![0u8; 32 - script_bytes.len()]);
            }
            vin["coinbase"] = serde_json::Value::String(hex::encode(script_bytes));
        }
    }
}

// Build coinbase with automatic BIP34 block height insertion
pub fn build_coinbase_from_template_with_height(template: &serde_json::Value) -> serde_json::Value {
    let mut coinbase = template.clone();
    let height = template["result"]["height"].as_u64().unwrap_or(0) as u32;

    let height_bytes = height.to_le_bytes();
    let height_prefix = vec![height_bytes.len() as u8];
    let mut coinbase_script = height_prefix;
    coinbase_script.extend_from_slice(&height_bytes);

    if let Some(tx) = coinbase["result"]["transactions"].get_mut(0) {
        if let Some(vin) = tx["vin"].get_mut(0) {
            let prev_script = hex::decode(vin["coinbase"].as_str().unwrap_or("")).unwrap_or_default();
            let mut final_script = coinbase_script;
            final_script.extend(prev_script);
            vin["coinbase"] = serde_json::Value::String(hex::encode(final_script));
        }
    }

    coinbase
}
