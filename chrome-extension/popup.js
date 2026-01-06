// PVTT Data Collector - Popup Script

console.log('Popup script started');

let currentData = null;

// 页面加载时执行
document.addEventListener('DOMContentLoaded', async () => {
  console.log('DOMContentLoaded event fired');

  const loading = document.getElementById('loading');
  const content = document.getElementById('content');

  console.log('Loading element:', loading);
  console.log('Content element:', content);

  try {
    // 获取当前标签页
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    console.log('Current URL:', tab.url);

    // 检查是否是 Etsy 商品页
    if (!tab.url.includes('etsy.com/listing/')) {
      showStatus('❌ Not a listing page. Please open a product page like:\nhttps://www.etsy.com/listing/123456/product-name', 'error');
      loading.style.display = 'none';
      return;
    }

    // 先尝试注入 content script（如果已经注入会失败但不影响）
    console.log('Injecting content script...');
    loading.textContent = 'Loading...';

    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js']
      });
      console.log('Content script injected');
    } catch (err) {
      console.log('Script injection failed (probably already injected):', err.message);
      // 忽略错误，可能已经注入过了
    }

    // 等待一下让 script 初始化
    await new Promise(resolve => setTimeout(resolve, 300));

    // 向 content script 请求数据
    console.log('Sending message to content script...');

    const handleResponse = (response) => {
      console.log('Received response:', response);

      if (chrome.runtime.lastError) {
        console.error('Runtime error:', chrome.runtime.lastError);
        showStatus('Error: ' + chrome.runtime.lastError.message, 'error');
        loading.style.display = 'none';
        return;
      }

      if (!response) {
        console.error('No response from content script');
        showStatus('No data received. Please refresh the page.', 'error');
        loading.style.display = 'none';
        return;
      }

      currentData = response;
      displayData(response);

      loading.style.display = 'none';
      content.style.display = 'block';
    };

    chrome.tabs.sendMessage(tab.id, { action: 'extractMedia' }, handleResponse);
  } catch (error) {
    showStatus('Error: ' + error.message, 'error');
    loading.style.display = 'none';
  }
});

// 显示数据
function displayData(data) {
  document.getElementById('listingId').textContent = data.listingId || 'Unknown';
  document.getElementById('title').textContent = data.title || 'No title';

  // 视频列表
  const videoList = document.getElementById('videoList');
  const videoCount = document.getElementById('videoCount');
  videoCount.textContent = data.videos.length;

  if (data.videos.length > 0) {
    videoList.innerHTML = data.videos.map((url, i) =>
      `<div class="list-item">${i + 1}. ${url.substring(0, 50)}...</div>`
    ).join('');
  } else {
    videoList.innerHTML = '<div class="list-item">No videos found</div>';
  }

  // 图片列表
  const imageList = document.getElementById('imageList');
  const imageCount = document.getElementById('imageCount');
  imageCount.textContent = data.images.length;

  if (data.images.length > 0) {
    imageList.innerHTML = data.images.slice(0, 5).map((url, i) =>
      `<div class="list-item">${i + 1}. ${url.substring(0, 50)}...</div>`
    ).join('');
    if (data.images.length > 5) {
      imageList.innerHTML += `<div class="list-item">...and ${data.images.length - 5} more</div>`;
    }
  } else {
    imageList.innerHTML = '<div class="list-item">No images found</div>';
  }

  // 显示 Etsy 完整分类路径（忠实记录）
  if (data.breadcrumbs && data.breadcrumbs.length > 0) {
    document.getElementById('breadcrumbs').textContent = data.breadcrumbs.join(' > ');
    document.getElementById('breadcrumbSection').style.display = 'block';
    console.log('Etsy Taxonomy:', data.breadcrumbs);
  }
}

// 下载按钮
document.getElementById('downloadBtn').addEventListener('click', async () => {
  // 全面验证数据完整性
  if (!currentData) {
    showStatus('No data available', 'error');
    return;
  }

  if (!currentData.listingId) {
    showStatus('Missing listing ID', 'error');
    return;
  }

  if (!currentData.productHandle) {
    showStatus('Missing product handle', 'error');
    return;
  }

  if (!currentData.videos || currentData.videos.length === 0) {
    showStatus('No video to download', 'error');
    return;
  }

  if (!currentData.breadcrumbs || currentData.breadcrumbs.length === 0) {
    showStatus('No category information found', 'error');
    return;
  }

  if (!currentData.title) {
    showStatus('Missing product title', 'error');
    return;
  }

  showStatus('Downloading...', 'info');

  try {
    // 发送到 background script 处理下载
    chrome.runtime.sendMessage({
      action: 'downloadSample',
      data: currentData
    }, (response) => {
      if (chrome.runtime.lastError) {
        showStatus('Error: ' + chrome.runtime.lastError.message, 'error');
        return;
      }

      if (response && response.success) {
        showStatus(`✓ Downloaded listing ${response.listingId}`, 'success');
      } else {
        showStatus('Error: ' + (response?.error || 'Unknown error'), 'error');
      }
    });
  } catch (error) {
    showStatus('Error: ' + error.message, 'error');
  }
});

// 复制按钮
document.getElementById('copyBtn').addEventListener('click', () => {
  if (!currentData) {
    showStatus('No data to copy', 'error');
    return;
  }

  const text = JSON.stringify(currentData, null, 2);
  navigator.clipboard.writeText(text).then(() => {
    showStatus('✓ Copied to clipboard', 'success');
  });
});

// 显示状态消息
function showStatus(message, type = 'info') {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = `status ${type}`;

  if (type === 'success' || type === 'info') {
    setTimeout(() => {
      status.style.display = 'none';
    }, 3000);
  }
}
