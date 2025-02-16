# 100try19_realtimedetect

## プロジェクトの概要
このプロジェクトは、Webカメラを使用してリアルタイムで姿勢推定や手のジェスチャー検出を行うWebアプリケーションです。
ブラウザ上でカメラからの映像をリアルタイムで表示し、MediaPipeを使用して検出を行います。

## 主な機能
- Webカメラからのリアルタイム映像取得
- 検出結果の表示

## 使用技術
- **フレームワーク**: Flask（Python）
- **フロントエンド**: HTML, CSS, JavaScript
- **画像処理**: OpenCV
- **検出エンジン**: MediaPipe
- **カメラ制御**: WebRTC

## 環境構築手順
1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

2. アプリケーションの起動
```bash
python app.py
```

3. ブラウザでアクセス
```
http://localhost:5000
```

## ファイル構成
```
プロジェクトフォルダ
├── README.md           # 本ドキュメント
├── app.py             # メインアプリケーション
├── requirements.txt   # 依存パッケージリスト
├── static/           # 静的ファイル
│   └── styles.css    # スタイルシート
└── templates/        # テンプレート
    └── index.html    # メインページ
```

## 注意事項
- アプリケーションの使用にはWebカメラが必要です
- ブラウザでのカメラへのアクセス許可が必要です
- 十分な処理能力を持つPCでの実行を推奨します

## ライセンス
MITライセンス
