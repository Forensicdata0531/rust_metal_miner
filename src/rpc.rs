use reqwest::Client;
use serde_json::Value;

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
