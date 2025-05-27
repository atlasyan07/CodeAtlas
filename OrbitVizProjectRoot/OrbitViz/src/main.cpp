#include "MainWindow.h"
#include <QApplication>
#include <QSurfaceFormat>
#include <QDir>
#include <QDebug>

int main(int argc, char *argv[])
{
    // Configure OpenGL format
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setVersion(3, 3);  // Request OpenGL 3.3 core profile
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(0);     // Disable MSAA for software rendering
    QSurfaceFormat::setDefaultFormat(format);

    // Enable high DPI scaling
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    QApplication app(argc, argv);

    // Print OpenGL information
    qDebug() << "QSurfaceFormat default format:" << QSurfaceFormat::defaultFormat().renderableType();
    qDebug() << "OpenGL format version:" 
             << QSurfaceFormat::defaultFormat().majorVersion() << "."
             << QSurfaceFormat::defaultFormat().minorVersion();

    // Create default data directory structure if it doesn't exist
    QDir appDir = QCoreApplication::applicationDirPath();
    
    QDir dataDir(appDir.absolutePath() + "/data");
    if (!dataDir.exists()) {
        dataDir.mkpath(".");
        dataDir.mkpath("mock");
    }
    
    QDir resourcesDir(appDir.absolutePath() + "/resources");
    if (!resourcesDir.exists()) {
        resourcesDir.mkpath(".");
        resourcesDir.mkpath("models");
        resourcesDir.mkpath("textures");
    }

    // Create and show the main window with error handling
    try {
        // Create and show the main window
        MainWindow mainWindow;
        mainWindow.resize(1280, 800);  // Larger default size for better visualization
        mainWindow.show();

        return app.exec();
    } catch (const std::exception& e) {
        qCritical() << "Exception in main:" << e.what();
        return 1;
    } catch (...) {
        qCritical() << "Unknown exception in main";
        return 1;
    }
}