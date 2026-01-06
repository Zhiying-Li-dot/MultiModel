// PVTT Data Collector - Background Script
// 处理下载和数据保存

// 监听来自 popup 的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'downloadSample') {
    downloadSample(request.data)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // 保持消息通道开启
  }
});

async function downloadSample(data) {
  try {
    // 验证必需字段
    if (!data.listingId || !data.productHandle) {
      return { success: false, error: 'Missing listing ID or product handle' };
    }

    if (!data.videos || data.videos.length === 0) {
      return { success: false, error: 'No videos found for this product' };
    }

    // 从 breadcrumbs 获取分类信息
    const etsyTaxonomy = data.breadcrumbs || [];
    if (etsyTaxonomy.length === 0) {
      return { success: false, error: 'No category information' };
    }

    // 生成下载路径：PVTT/{date}/{listingId_productHandle}/
    const now = new Date();
    const date = now.toISOString().split('T')[0]; // 格式：2026-01-06

    // 清理 productHandle 中的文件系统非法字符
    const sanitizedHandle = data.productHandle.replace(/[/\\?%*:|"<>]/g, '-');
    const folderName = `${data.listingId}_${sanitizedHandle}`;
    const basePath = `PVTT/${date}/${folderName}`;

    // 下载视频
    const videoFilename = `video.mp4`;
    await downloadFile(data.videos[0], `${basePath}/${videoFilename}`);

    // 下载所有图片
    const imageFilenames = [];
    for (let i = 0; i < data.images.length; i++) {
      const url = data.images[i];
      // 更健壮的文件扩展名检测
      let ext = '.jpg'; // 默认
      if (url.includes('.webp')) ext = '.webp';
      else if (url.includes('.png')) ext = '.png';
      else if (url.includes('.jpeg')) ext = '.jpeg';
      else if (url.includes('.gif')) ext = '.gif';

      const filename = `image_${i + 1}${ext}`;
      await downloadFile(url, `${basePath}/${filename}`);
      imageFilenames.push(filename);
    }

    // 创建当前商品的 metadata（忠实记录产品信息）
    const metadata = {
      listing_id: data.listingId,
      product_handle: data.productHandle,
      url: data.url,
      title: data.title,
      etsy_taxonomy: etsyTaxonomy,
      download_date: now.toISOString(),
      video: {
        filename: videoFilename,
        source_url: data.videos[0]
      },
      images: imageFilenames.map((filename, i) => ({
        filename: filename,
        source_url: data.images[i]
      }))
    };

    // 导出 metadata.json 到商品文件夹（使用 Data URL）
    const metadataJson = JSON.stringify(metadata, null, 2);
    const metadataDataUrl = 'data:application/json;charset=utf-8,' + encodeURIComponent(metadataJson);
    await chrome.downloads.download({
      url: metadataDataUrl,
      filename: `${basePath}/metadata.json`,
      conflictAction: 'overwrite',  // 覆盖已存在的文件
      saveAs: false
    });

    return { success: true, listingId: data.listingId };
  } catch (error) {
    console.error('Download error:', error);
    return { success: false, error: error.message };
  }
}

async function downloadFile(url, filename) {
  return new Promise((resolve, reject) => {
    chrome.downloads.download({
      url: url,
      filename: filename,
      conflictAction: 'overwrite',  // 覆盖已存在的文件
      saveAs: false
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(downloadId);
      }
    });
  });
}

console.log('PVTT Data Collector: Background script loaded');
