#include "mainwindow.h"
#include <QApplication>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), currentModelPath("./gpt2_124M100Steps.bin")  // Default path
{
    setupUi();
    applyStyles();
}

MainWindow::~MainWindow()
{
    // Add cleanup code here if needed
}

void MainWindow::setupUi()
{
    // Set window properties
    setMinimumSize(800, 600);
    setWindowTitle("NanoInference");

    // Create main widgets
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    // Create vertical layout for the entire window
    QVBoxLayout* mainVLayout = new QVBoxLayout(centralWidget);
    mainVLayout->setSpacing(0);
    mainVLayout->setContentsMargins(0, 0, 0, 0);

    // Create and add header
    createHeader();
    mainVLayout->addWidget(headerWidget);

    // Create horizontal layout for sidebar and chat area
    QHBoxLayout* contentLayout = new QHBoxLayout();
    contentLayout->setSpacing(0);
    contentLayout->setContentsMargins(0, 0, 0, 0);

    // Create and add sidebar and chat area
    createSidebar();
    createChatArea();
    contentLayout->addWidget(sidebar);
    contentLayout->addWidget(chatArea);

    // Connect signals
    connect(sendButton, &QPushButton::clicked, this, &MainWindow::onSendClicked);
    connect(loadModelBtn, &QPushButton::clicked, this, &MainWindow::onLoadModelClicked);

    // Add content layout to main layout
    mainVLayout->addLayout(contentLayout);

    // Create and add status bar
    createStatusBar();
}

void MainWindow::createHeader()
{
    headerWidget = new QWidget();
    headerWidget->setFixedHeight(60);
    headerWidget->setObjectName("headerWidget");

    QHBoxLayout* headerLayout = new QHBoxLayout(headerWidget);
    headerLayout->setContentsMargins(20, 0, 20, 0);
    
    logoWidget = new QLabel();
    
    // Debug available resources
    qDebug() << "Available resource paths:";
    QDirIterator it(":", QDirIterator::Subdirectories);
    while (it.hasNext()) {
        qDebug() << it.next();
    }
    
    // Load the PNG image from resources
    QString logoPath = ":/images/cstar-logo.png";
    qDebug() << "Attempting to load logo from:" << logoPath;
    QPixmap logo(logoPath);
    
    if (logo.isNull()) {
        qDebug() << "Failed to load logo from resources";
        logoWidget->setText("CSTAR");  // Fallback text
    } else {
        qDebug() << "Logo loaded successfully, dimensions:" << logo.width() << "x" << logo.height();
        logo = logo.scaledToHeight(50, Qt::SmoothTransformation);
        logoWidget->setPixmap(logo);
    }
    
    // Center the logo vertically in the header
    headerLayout->addWidget(logoWidget, 0, Qt::AlignVCenter);
    
    // Add a stretching spacer to push the logo to the left
    headerLayout->addStretch(1);
}

void MainWindow::createSidebar()
{
    sidebar = new QWidget();
    sidebar->setFixedWidth(230);
    sidebarLayout = new QVBoxLayout(sidebar);
    sidebarLayout->setContentsMargins(20, 20, 20, 20);
    
    // Model Selection Group
    QLabel* modelLabel = new QLabel("Model Selection");
    modelComboBox = new QComboBox();
    
    // Configure the combo box for display-only with proper styling
    modelComboBox->setEditable(false);  // Not editable
    modelComboBox->setToolTip("Currently loaded model");
    
    // Add initial items - these will be example models or recently used models
    modelComboBox->addItem("GPT-2-small");
    modelComboBox->addItem("GPT-2-medium");
    modelComboBox->addItem("GPT-2-large");
    
    // Display the currently loaded model filename
    updateModelDisplay();
    
    loadModelBtn = new QPushButton("Load Model");
    
    // Settings Group
    QLabel* settingsLabel = new QLabel("Settings");
    tokenLabel = new QLabel("Approx words output:");
    tokenSpinBox = new QSpinBox();
    tokenSpinBox->setRange(1,1024);
    tokenSpinBox->setValue(20);  // Default value
    
    // Advanced Settings Group
    QLabel* advancedSettingsLabel = new QLabel("Advanced Settings");
    QLabel* temperatureLabel = new QLabel("Temperature:");
    temperatureSpinBox = new QDoubleSpinBox();
    temperatureSpinBox->setRange(0.1, 2.0);
    temperatureSpinBox->setValue(1.0);  // Default value
    temperatureSpinBox->setSingleStep(0.1);
    
    // Add widgets to sidebar
    sidebarLayout->addWidget(modelLabel);
    sidebarLayout->addWidget(modelComboBox);
    sidebarLayout->addWidget(loadModelBtn);
    sidebarLayout->addSpacing(20);
    sidebarLayout->addWidget(settingsLabel);
    sidebarLayout->addWidget(tokenLabel);
    sidebarLayout->addWidget(tokenSpinBox);
    sidebarLayout->addSpacing(20);
    sidebarLayout->addWidget(advancedSettingsLabel);
    sidebarLayout->addWidget(temperatureLabel);
    sidebarLayout->addWidget(temperatureSpinBox);
    sidebarLayout->addStretch();

    // Set object names for styling
    sidebar->setObjectName("sidebar");
    loadModelBtn->setObjectName("loadModelBtn");

    // Make model label more prominent
    modelLabel->setStyleSheet("font-size: 14px; margin-bottom: 5px; color: #121212;");
    
    // Style the settings section with more spacing
    settingsLabel->setStyleSheet("font-size: 14px; margin-top: 5px; margin-bottom: 5px; color: #121212;");
    
    // Style the advanced settings section
    advancedSettingsLabel->setStyleSheet("font-size: 14px; margin-top: 5px; margin-bottom: 5px; color: #121212;");
}

void MainWindow::updateModelDisplay()
{
    // Extract filename for display
    QFileInfo fileInfo(currentModelPath);
    QString fileName = fileInfo.fileName();
    
    // Check if the model filename is already in the combobox
    int index = modelComboBox->findText(fileName);
    
    if (index >= 0) {
        // If found, select it
        modelComboBox->setCurrentIndex(index);
    } else {
        // If not found, add it and select it
        modelComboBox->insertItem(0, fileName);
        modelComboBox->setCurrentIndex(0);
    }
    
    // Set tooltip to show full path on hover
    modelComboBox->setToolTip(currentModelPath);
}

void MainWindow::createChatArea()
{
    chatArea = new QWidget();
    chatLayout = new QHBoxLayout(chatArea);  // Changed from QVBoxLayout to QHBoxLayout
    chatLayout->setContentsMargins(20, 20, 20, 20);
    
    // Create input container (middle column)
    QWidget* inputContainer = new QWidget();
    QVBoxLayout* inputLayout = new QVBoxLayout(inputContainer);
    inputLayout->setContentsMargins(10, 0, 10, 0);
    
    // Add a label for the input section
    QLabel* inputLabel = new QLabel("Input");
    inputLabel->setStyleSheet("font-size: 14px; font-weight: bold; color: #E5E7EB;");
    
    messageInput = new QTextEdit();
    messageInput->setMinimumHeight(300);  // Give it more height
    messageInput->setPlaceholderText("Type your message here...");
    
    sendButton = new QPushButton("Send");
    sendButton->setFixedHeight(50);
    sendButton->setObjectName("sendButton");
    
    inputLayout->addWidget(inputLabel);
    inputLayout->addWidget(messageInput, 1);
    inputLayout->addWidget(sendButton);
    
    // Create output container (right column)
    QWidget* outputContainer = new QWidget();
    QVBoxLayout* outputLayout = new QVBoxLayout(outputContainer);
    outputLayout->setContentsMargins(10, 0, 0, 0);
    
    // Add a label for the output section
    QLabel* outputLabel = new QLabel("Output");
    outputLabel->setStyleSheet("font-size: 14px; font-weight: bold; color: #E5E7EB;");
    
    chatDisplay = new QTextEdit();
    chatDisplay->setReadOnly(true);
    chatDisplay->setObjectName("chatDisplay");
    
    outputLayout->addWidget(outputLabel);
    outputLayout->addWidget(chatDisplay);
    
    // Add both containers to the main chat layout
    chatLayout->addWidget(inputContainer, 1);  // 1:1 ratio with output
    chatLayout->addWidget(outputContainer, 1);
}



void MainWindow::onSendClicked() {
    QString message = messageInput->toPlainText().trimmed();
    if (message.isEmpty()) return;
    
    // Show "thinking" indicator
    chatDisplay->append("<i>Thinking...</i>");

    // Get max tokens from spinbox
    int maxTokens = (tokenSpinBox->value())/0.75;  // Approximate words to tokens
    maxTokens = maxTokens > 1024 ? 1024 : maxTokens;  // Cap at 1024 tokens

    // Move cursor to end of chat
    QScrollBar *scrollbar = chatDisplay->verticalScrollBar();
    scrollbar->setValue(scrollbar->maximum());

    // Process in a background thread
    QFuture<QString> future = QtConcurrent::run([this, message, maxTokens]() {
        return runInferenceProcess(message, maxTokens);
    });

    // Connect to watcher to update UI when done
    QFutureWatcher<QString>* watcher = new QFutureWatcher<QString>(this);
    connect(watcher, &QFutureWatcher<QString>::finished, this, [this, watcher]() {
        QString response = watcher->result();
        
        // Remove thinking indicator
        QString currentText = chatDisplay->toHtml();
        int lastParagraph = currentText.lastIndexOf("<p");
        if (lastParagraph > 0 && currentText.indexOf("Thinking...", lastParagraph) > 0) {
            currentText = currentText.left(lastParagraph);
            chatDisplay->setHtml(currentText);
        }
        
        // Add model response
        chatDisplay->append(response);
        
        // Scroll to bottom
        QScrollBar *scrollbar = chatDisplay->verticalScrollBar();
        scrollbar->setValue(scrollbar->maximum());
        
        watcher->deleteLater();
    });
    
    watcher->setFuture(future);
}

QString MainWindow::runInferenceProcess(const QString& prompt, int maxTokens) {
    // Create a process to run the inference CLI
    QProcess process;
    
    // Set the working directory to where the model files are
    process.setWorkingDirectory(QCoreApplication::applicationDirPath());
    
    // Prepare the command and arguments
    QString program = QCoreApplication::applicationDirPath() + "/inference_cli";
    QStringList arguments;

    // Get the temperature value
    double temperature = temperatureSpinBox->value();
    temperature = temperature <0.0 ? 0.0 : temperature;  //Minimum value
    
    // Use the current model path - pass as extra argument to inference_cli
    arguments << "--generate" << prompt << QString::number(maxTokens) 
              << "--temperature" << QString::number(temperature) << currentModelPath;
    
    // Start the process
    process.start(program, arguments);
    
    // Wait for the process to finish with a timeout
    if (!process.waitForFinished(300000)) { // 300 second timeout
        process.kill();
        return "Error: Inference process timed out.";
    }
    
    // Check for errors
    if (process.exitCode() != 0) {
        QString errorOutput = process.readAllStandardError();
        return "Error: " + errorOutput;
    }
    
    // Read the output
    QString output = process.readAllStandardOutput();
    return output;
}

void MainWindow::onLoadModelClicked() {
    // Show file dialog to select model weights file
    QString modelPath = QFileDialog::getOpenFileName(this,
        tr("Select Model Weights File"),
        QCoreApplication::applicationDirPath(),
        tr("Model Files (*.bin *.pt);;All Files (*)"));
    
    // If user canceled the dialog, return without doing anything
    if (modelPath.isEmpty()) {
        return;
    }
    
    // Update UI to show loading state
    loadModelBtn->setEnabled(false);
    loadModelBtn->setText("Loading...");
    modelStatusLabel->setText("Model Status: Loading...");
    
    // Extract filename for display
    QFileInfo fileInfo(modelPath);
    QString fileName = fileInfo.fileName();
    
    // Run in background
    QFuture<bool> future = QtConcurrent::run([this, modelPath]() {
        return loadModelProcess(modelPath);
    });
    
    // Set up watcher
    QFutureWatcher<bool>* watcher = new QFutureWatcher<bool>(this);
    connect(watcher, &QFutureWatcher<bool>::finished, this, [this, watcher, fileName, modelPath]() {
        bool success = watcher->result();
        
        if (success) {
            // Store the successful model path
            currentModelPath = modelPath;

            // Update the model display in the UI
            updateModelDisplay();
            
            loadModelBtn->setText("Change Model");
            loadModelBtn->setEnabled(true);
            modelStatusLabel->setText("Model Active: " + fileName);
        } else {
            loadModelBtn->setEnabled(true);
            loadModelBtn->setText("Load Model");
            modelStatusLabel->setText("Model Status: Failed to load");
            chatDisplay->append("<span style='color:red'>Error: Failed to load model " + fileName + ".</span>");
        }
        
        watcher->deleteLater();
    });
    
    watcher->setFuture(future);
}

bool MainWindow::loadModelProcess(const QString& modelPath) {
    // Create a process to run the inference CLI with load command
    QProcess process;
    
    // Set the working directory
    process.setWorkingDirectory(QCoreApplication::applicationDirPath());
    
    // Prepare the command
    QString program = QCoreApplication::applicationDirPath() + "/inference_cli";
    QStringList arguments;
    arguments << "--load" << modelPath;
    
    // Start the process
    process.start(program, arguments);
    
    // Wait for the process to finish with a timeout
    if (!process.waitForFinished(30000)) { // 30 second timeout
        process.kill();
        return false;
    }
    
    // Check for success
    return process.exitCode() == 0;
}

void MainWindow::createStatusBar()
{
    QStatusBar* statusBar = new QStatusBar();
    modelStatusLabel = new QLabel("Model Active: None");
    memoryStatusLabel = new QLabel("Memory: 0GB / 8GB");
    
    statusBar->addWidget(modelStatusLabel);
    statusBar->addPermanentWidget(memoryStatusLabel);
    setStatusBar(statusBar);
}

void MainWindow::applyStyles()
{
    // Apply styles using stylesheets - professional dark mode color scheme
    setStyleSheet(R"(
        QMainWindow {
            background-color: #121212;
        }
        
        /* Header with darker background */
        QWidget#headerWidget {
            background-color: #121212;
            border-bottom: 1px solid #374151;
        }
        
        /* Sidebar with slightly lighter background for contrast */
        QWidget#sidebar {
            background-color: #1E1E1E;
            border-right: 1px solid #374151;
        }
        
        /* Light text for sidebar labels */
        QWidget#sidebar QLabel {
            color: #E5E7EB;
            font-weight: bold;
        }
        
        /* Buttons with accent color and hover effects */
        QPushButton#loadModelBtn {
            background-color: #3B82F6;
            color: #E5E7EB;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
        }
        
        QPushButton#loadModelBtn:hover {
            background-color: #4B93FF;
        }
        
        QPushButton#loadModelBtn:pressed {
            background-color: #2970E3;
        }
        
        /* Chat display with dark background and light text */
        QTextEdit#chatDisplay {
            background-color: #1E1E1E;
            border: 1px solid #374151;
            border-radius: 6px;
            color: #E5E7EB;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        
        /* Rounded send button with accent color */
        QPushButton#sendButton {
            background-color:rgb(133, 149, 173);
            color: #E5E7EB;
            border-radius: 25px;
            font-weight: bold;
        }
        
        QPushButton#sendButton:hover {
            background-color: #4B93FF;
        }
        
        QPushButton#sendButton:pressed {
            background-color: #2970E3;
        }
        
        /* Status bar with subtle top border */
        QStatusBar {
            background-color: #121212;
            border-top: 1px solid #374151;
            color: #9CA3AF;
        }
        
        /* Text inputs with dark styling */
        QTextEdit {
            color: #E5E7EB;
            background-color: #1E1E1E;
            border: 1px solid #374151;
            border-radius: 4px;
            selection-background-color: #8B5CF6;
        }
        
        /* ComboBox styling */
        QComboBox {
            border: 1px solid #374151;
            border-radius: 3px;
            padding: 3px 15px 3px 5px;
            background-color: #1E1E1E;
            color: #E5E7EB;
        }
        
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 20px;
            border-left: 1px solid #374151;
            background-color: #272727;
        }
        
        /* SpinBox styling */
        QSpinBox {
            border: 1px solid #374151;
            border-radius: 3px;
            padding: 3px;
            background-color: #1E1E1E;
            color: #E5E7EB;
        }
        
        QDoubleSpinBox {
            border: 1px solid #374151;
            border-radius: 3px;
            padding: 3px;
            background-color: #1E1E1E;
            color: #E5E7EB;
        }
        
        /* Label in header - in case logo fails to load */
        QLabel#logoWidget {
            color: #E5E7EB;
            font-size: 22px;
            font-weight: bold;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        /* Scrollbar styling for a modern look */
        QScrollBar:vertical {
            border: none;
            background: #1E1E1E;
            width: 8px;
            margin: 0px;
        }

        QScrollBar::handle:vertical {
            background: #374151;
            min-height: 20px;
            border-radius: 4px;
        }

        QScrollBar::handle:vertical:hover {
            background: #4B5563;
        }

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }

        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        
        /* Success and error message styling */
        QLabel[style='success'] {
            color: #22C55E;
        }
        
        QLabel[style='error'] {
            color: #EF4444;
        }
    )");
    
    // Fix the styling for section labels in the sidebar to match dark theme
    QList<QLabel*> labels = sidebar->findChildren<QLabel*>();
    for (QLabel* label : labels) {
        if (label->text() == "Model Selection" || 
            label->text() == "Settings" || 
            label->text() == "Advanced Settings") {
            label->setStyleSheet("font-size: 14px; margin-bottom: 5px; color: #E5E7EB;");
        }
    }
}