package ddprofiler;

import java.io.IOException;
import java.util.List;
import java.util.Properties;

import org.junit.Test;

import ddprofiler.core.Main;
import ddprofiler.core.config.CommandLineArgs;
import ddprofiler.core.config.ConfigKey;
import ddprofiler.core.config.ProfilerConfig;

import joptsimple.OptionParser;

public class TestOfflineMode {

    public void proceedCommand(String[] command) {
        List<ConfigKey> configKeys = ProfilerConfig.getAllConfigKey();
        OptionParser parser = new OptionParser();
        // Unrecognized options are passed through to the query
        parser.allowsUnrecognizedOptions();
        CommandLineArgs cla = new CommandLineArgs(command, parser, configKeys);
        Properties commandLineProperties = cla.getProperties();
        Properties validatedProperties =
                Main.validateProperties(commandLineProperties);

        ProfilerConfig pc = new ProfilerConfig(validatedProperties);
        if (validatedProperties.getProperty("help") != null) {
            pc.toString();
        }

        // Start main
        Main m = new Main();
        try {
            m.startProfiler(pc, parser);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void offlineModeTest() {
        String[] args_test1 = new String[]{"-execution.mode",
                "1",
                "-num.record.read",
                "100",
                "-sources.folder.path",
                "C:\\csv\\",
                "-help"};
        proceedCommand(args_test1);
    }
}
