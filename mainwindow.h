#pragma once

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include <QTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QLabel>
#include <QComboBox>
#include <QStatusBar>
#include <QString>
#include <QScrollBar>
#include <QtConcurrent>
#include <QProcess>
#include <QTemporaryFile>
#include <QFileDialog>
#include <QDoubleSpinBox>

//#include <QLineEdit>


class InferenceInterface {
    public:
        static bool isModelLoaded() {
            return modelLoaded;
        }
        
        static void setModelLoaded(bool loaded) {
            modelLoaded = loaded;
        }
        
        static QString generateResponse(const QString& input, int maxTokens);
        static bool loadModel();
        
    private:
        static bool modelLoaded;
    };

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    void setupUi();
    void createHeader();
    void createSidebar();
    void createChatArea();
    void createStatusBar();
    void applyStyles();
    QString runInferenceProcess(const QString& prompt, int maxTokens);
    bool loadModelProcess(const QString& modelPath);
    void updateModelDisplay();

private:
    // Header widgets
    QWidget* headerWidget;
    QLabel* logoWidget;

    // Central widget and main layout
    QWidget* centralWidget;
    QHBoxLayout* mainLayout;
    
    // Sidebar widgets
    QWidget* sidebar;
    QVBoxLayout* sidebarLayout;
    QComboBox* modelComboBox;
    QPushButton* loadModelBtn;
    QLabel* tokenLabel;
    QSpinBox* tokenSpinBox;
    QDoubleSpinBox* temperatureSpinBox;

    // Chat area widgets
    QWidget* chatArea;
    QHBoxLayout* chatLayout;
    QTextEdit* chatDisplay;
    QTextEdit* messageInput;
    QPushButton* sendButton;

    // Status bar widgets
    QLabel* modelStatusLabel;
    QLabel* memoryStatusLabel;

    QString currentModelPath;  // Store the path to the currently loaded model

private slots:
    void onSendClicked();
    void onLoadModelClicked();
    

};