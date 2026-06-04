### SLA - SLO - SLI

超好的例子：

    假設你是某個線上平台的 PM，和使用者約定 SLA 是「每個月不能有超過 40 分鐘的中斷時間」。

    SLI：這個月實際的系統可用率是 99.96%，有 17 分鐘中斷。
    SLO：我們內部要求可用率達 99.95%。
    SLA：客戶 SLA 是 99.9%，你達到了，無需賠償。

🎯 你可以用這三個數據來跟主管說：**「本月服務穩定，SLI 落在 SLO 內，也符合 SLA。」**

https://vocus.cc/article/68007b4efd89780001df4e98

- SLA（Service Level Agreement）：對「客戶」的服務保證
    - 保證 99.9% 可用性，超過要賠錢（合約）
- SLO（Service Level Objective）:對「內部團隊」的目標
    - 我們團隊內部定的標準，例如「99.95% 可用性」
- SLI（Service Level Indicator）:量測服務狀況的數字
    - 實際可用性、延遲、錯誤率的統計指標
