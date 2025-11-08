use reqwest::Client;
use serde_json::Value;
use bitcoin::blockdata::block::{BlockHeader, Block as BtcBlock};
use bitcoin::hashes::sha256d;
use bitcoin::hash_types::TxMerkleNode;
use bitcoin::consensus::encode::serialize;
use bitcoin::hashes::Hash;
use hex;
use std::str::FromStr;

use crate::sha_helpers::{
    double_sha256_bytes, target_from_bits, hash_le_target, serialize_block_header_bytes,
};
use crate::coinbase::{build_coinbase_from_template, insert_nonce_into_coinbase, assemble_block_hex};

/// Fetches current block template from Bitcoin Core.
pub async fn fetch_block_template(
    client: &Client,
    url: &str,
    user: &str,
    pass: &str,
) -> Option<Value> {
    let res = client
        .post(url)
        .basic_auth(user, Some(pass))
        .json(&serde_json::json!({
            "jsonrpc": "1.0",
            "id": "rustminer",
            "method": "getblocktemplate",
            "params": [{"rules":["segwit"]}]
        }))
        .send()
        .await
        .ok()?;
    res.json().await.ok()
}

/// Submits a raw block hex to Bitcoin Core.
pub async fn submit_block(
    client: &Client,
    user: &str,
    pass: &str,
    block_hex: &str,
) -> Result<(), reqwest::Error> {
    let res = client
        .post("http://127.0.0.1:8332")
        .basic_auth(user, Some(pass))
        .json(&serde_json::json!({
            "jsonrpc":"1.0",
            "id":"rustminer",
            "method":"submitblock",
            "params":[block_hex]
        }))
        .send()
        .await?;

    if res.status().is_success() {
        println!("✅ Block submitted successfully");
    } else {
        println!("❌ Block submission failed: {:?}", res.text().await);
    }
    Ok(())
}

/// Full validation + submit wrapper.
/// Builds the block with nonce, computes double-SHA256 of header, compares to target,
/// and calls `submitblock` only if valid.
pub async fn try_and_submit_nonce(
    client: &Client,
    rpc_user: &str,
    rpc_pass: &str,
    template: &Value,
    coinbase_tx: &bitcoin::Transaction,
    nonce: u32,
) -> Result<bool, reqwest::Error> {
    // --- Build header fields from template ---
    let version = template["result"]["version"].as_i64().unwrap_or(4) as i32;
    let prevhash_str = template["result"]["previousblockhash"].as_str().unwrap_or("00");
    let prevhash = sha256d::Hash::from_str(prevhash_str)
        .unwrap_or_else(|_| sha256d::Hash::hash(&[0u8; 32]));
    let bits_hex = template["result"]["bits"].as_str().unwrap_or("00000000");
    let bits = u32::from_str_radix(bits_hex, 16).unwrap_or(0u32);
    let time = template["result"]["curtime"].as_u64().unwrap_or(0) as u32;

    // --- Construct coinbase with nonce inserted ---
    let mut coinbase = coinbase_tx.clone();
    insert_nonce_into_coinbase(&mut coinbase, nonce);

    // --- Compute merkle root of all txs (coinbase + template txs) ---
    let mut txs = vec![coinbase.clone()];
    if let Some(txs_json) = template["result"]["transactions"].as_array() {
        for txj in txs_json {
            if let Some(data_str) = txj["data"].as_str() {
                if let Ok(raw) = hex::decode(data_str) {
                    if let Ok(tx) =
                        bitcoin::consensus::deserialize::<bitcoin::Transaction>(&raw)
                    {
                        txs.push(tx);
                    }
                }
            }
        }
    }

    let mut hashes: Vec<sha256d::Hash> =
        txs.iter().map(|t| t.txid().as_hash()).collect();
    while hashes.len() > 1 {
        if hashes.len() % 2 == 1 {
            let last = hashes.last().cloned().unwrap();
            hashes.push(last);
        }
        let mut new_hashes = Vec::with_capacity(hashes.len() / 2);
        for i in (0..hashes.len()).step_by(2) {
            let concat = [hashes[i].into_inner(), hashes[i + 1].into_inner()].concat();
            new_hashes.push(sha256d::Hash::hash(&concat));
        }
        hashes = new_hashes;
    }
    let merkle_root = TxMerkleNode::from_inner(hashes[0].into_inner());

    // --- Build header and compute hash ---
    let header = BlockHeader {
        version,
        prev_blockhash: prevhash.into(),
        merkle_root,
        time,
        bits,
        nonce,
    };
    let header_bytes = serialize_block_header_bytes(&header);
    let hash_be = double_sha256_bytes(&header_bytes);
    let target_be = target_from_bits(bits);

    // --- Compare ---
    if hash_le_target(&hash_be, &target_be) {
        let block_hex = assemble_block_hex(template, &coinbase, nonce);
        submit_block(client, rpc_user, rpc_pass, &block_hex).await?;
        println!("✅ Valid block! nonce = {nonce}");
        Ok(true)
    } else {
        Ok(false)
    }
}
