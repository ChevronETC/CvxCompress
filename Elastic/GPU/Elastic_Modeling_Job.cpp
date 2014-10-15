#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <omp.h>
#include "Elastic_Modeling_Job.hxx"
#include "Elastic_Shot.hxx"
#include "Voxet.hxx"
#include "Voxet_Property.hxx"
#include "Global_Coordinate_System.hxx"
#include "Elastic_Propagator.hxx"
#include "Elastic_SEGY_File.hxx"

Elastic_Modeling_Job::Elastic_Modeling_Job(
	int log_level,
	const char* parmfile_path
	)
{
	_Is_Valid = false;
	_log_level = log_level;
	_propagator = 0L;
	
	_num_em_props = 14;

	//Attr_Idx_Vp       = 0;
	//Attr_Idx_Vs       = 1;
	//Attr_Idx_Density  = 2;
	//Attr_Idx_Q        = 3;
	//Attr_Idx_Dip      = 4;
	//Attr_Idx_Azimuth  = 5;
	//Attr_Idx_Rake     = 6;
	//Attr_Idx_Delta1   = 7;
	//Attr_Idx_Delta2   = 8;
	//Attr_Idx_Delta3   = 9;
	//Attr_Idx_Epsilon1 = 10;
	//Attr_Idx_Epsilon2 = 11;
	//Attr_Idx_Gamma1   = 12;
	//Attr_Idx_Gamma2   = 13;

	_pck_moniker = new char*[_num_em_props];
	_pck_moniker[Attr_Idx_Vp]       = strdup("Vp");
	_pck_moniker[Attr_Idx_Vs]       = strdup("Vs");
	_pck_moniker[Attr_Idx_Density]  = strdup("Density");
	_pck_moniker[Attr_Idx_Q]        = strdup("Q");
	_pck_moniker[Attr_Idx_Dip]      = strdup("Dip");
	_pck_moniker[Attr_Idx_Azimuth]  = strdup("Azimuth");
	_pck_moniker[Attr_Idx_Rake]     = strdup("Rake");
	_pck_moniker[Attr_Idx_Delta1]   = strdup("Delta1");
	_pck_moniker[Attr_Idx_Delta2]   = strdup("Delta2");
	_pck_moniker[Attr_Idx_Delta3]   = strdup("Delta3");
	_pck_moniker[Attr_Idx_Epsilon1] = strdup("Epsilon1");
	_pck_moniker[Attr_Idx_Epsilon2] = strdup("Epsilon2");
	_pck_moniker[Attr_Idx_Gamma1]   = strdup("Gamma1");
	_pck_moniker[Attr_Idx_Gamma2]   = strdup("Gamma2");

	_pck_mask = new int[_num_em_props];
	_pck_mask[Attr_Idx_Vp]       = 65535;
	_pck_mask[Attr_Idx_Vs]       = 65535;
	_pck_mask[Attr_Idx_Density]  =   255;
	_pck_mask[Attr_Idx_Q]        =   255;
	_pck_mask[Attr_Idx_Dip]      =   255;
	_pck_mask[Attr_Idx_Azimuth]  =   255;
	_pck_mask[Attr_Idx_Rake]     =   255;
	_pck_mask[Attr_Idx_Delta1]   =   255;
	_pck_mask[Attr_Idx_Delta2]   =   255;
	_pck_mask[Attr_Idx_Delta3]   =   255;
	_pck_mask[Attr_Idx_Epsilon1] =   255;
	_pck_mask[Attr_Idx_Epsilon2] =   255;
	_pck_mask[Attr_Idx_Gamma1]   =   255;
	_pck_mask[Attr_Idx_Gamma2]   =   255;

	_pck_shft = new int[_num_em_props];
	_pck_shft[Attr_Idx_Vp]       = 16;
	_pck_shft[Attr_Idx_Vs]       =  0;
	_pck_shft[Attr_Idx_Density]  = 16;
	_pck_shft[Attr_Idx_Q]        = 24;
	_pck_shft[Attr_Idx_Dip]      =  8;
	_pck_shft[Attr_Idx_Azimuth]  =  0;
	_pck_shft[Attr_Idx_Rake]     = 24;
	_pck_shft[Attr_Idx_Delta1]   = 16;
	_pck_shft[Attr_Idx_Delta2]   =  8;
	_pck_shft[Attr_Idx_Delta3]   =  0;
	_pck_shft[Attr_Idx_Epsilon1] = 24;
	_pck_shft[Attr_Idx_Epsilon2] = 16;
	_pck_shft[Attr_Idx_Gamma1]   =  8;
	_pck_shft[Attr_Idx_Gamma2]   =  0;

	_pck_widx = new int[_num_em_props];
	_pck_widx[Attr_Idx_Vp]       = 0;
	_pck_widx[Attr_Idx_Vs]       = 0;
	_pck_widx[Attr_Idx_Density]  = 3;
	_pck_widx[Attr_Idx_Q]        = 3;
	_pck_widx[Attr_Idx_Dip]      = 3;
	_pck_widx[Attr_Idx_Azimuth]  = 3;
	_pck_widx[Attr_Idx_Rake]     = 1;
	_pck_widx[Attr_Idx_Delta1]   = 1;
	_pck_widx[Attr_Idx_Delta2]   = 1;
	_pck_widx[Attr_Idx_Delta3]   = 1;
	_pck_widx[Attr_Idx_Epsilon1] = 2;
	_pck_widx[Attr_Idx_Epsilon2] = 2;
	_pck_widx[Attr_Idx_Gamma1]   = 2;
	_pck_widx[Attr_Idx_Gamma2]   = 2;

	_pck_min = new float[_num_em_props];
	_pck_max = new float[_num_em_props];
	_pck_range = new float[_num_em_props];
	_pck_iso = new float[_num_em_props];
	// initialization of these tables is done after parameter file has been read

	bool error = false;
	_use_isotropic_sphere_during_source_injection = false;
	_Courant_Factor = 1.0f;
	_voxet = 0L;
	_shots = 0L;
	_num_shots = 0;
	_props = new Voxet_Property*[_num_em_props];
	_const_vals = new float[_num_em_props];
	for (int i = 0;  i < _num_em_props;  ++i)
	{
		_props[i] = 0L;
		_const_vals[i] = 0.0f;
	}
	_fq = 5.0;
	_sub_origin = 0;  // default is source
	_sub_x_set = false;
	_sub_y_set = false;
	_sub_z_set = false;
	_parm_sub_ix0 = 0;
	_parm_sub_ix1 = 0;
	_parm_sub_iy0 = 0;
	_parm_sub_iy1 = 0;
	_parm_sub_iz0 = 0;
	_parm_sub_iz1 = 0;
	_parm_nabc_sdx = 0;
	_parm_nabc_sdy = 0;
	_parm_nabc_top = 0;
	_parm_nabc_bot = 0;
	_parm_nabc_sdx_extend = false;
	_parm_nabc_sdy_extend = false;
	_parm_nabc_top_extend = false;
	_parm_nabc_bot_extend = false;
	_sub_ix0 = 0;
	_sub_ix1 = 0;
	_sub_iy0 = 0;
	_sub_iy1 = 0;
	_sub_iz0 = 0;
	_sub_iz1 = 0;
	_nabc_sdx = 0;
	_nabc_sdy = 0;
	_nabc_top = 0;
	_nabc_bot = 0;
	_nabc_sdx_extend = false;
	_nabc_sdy_extend = false;
	_nabc_top_extend = false;
	_nabc_bot_extend = false;
	_freesurface_enabled = true;
	_source_ghost_enabled = true;
	_receiver_ghost_enabled = true;
	_lower_Q_seafloor_enabled = false;
	_GPU_Devices = 0L;
	_num_GPU_Devices = 0;
	_GPU_Pipes = 0;
	_Steps_Per_GPU = 0;
	if (_log_level > 2) printf("Parameter file is %s.\n",parmfile_path);
	FILE* fp = fopen(parmfile_path, "r");
	if (fp != 0L)
	{
		int line_num = 0;
		char str[4096];
		for (char* s = fgets(str, 4096, fp);  s != 0L && !error;  s = fgets(str, 4096, fp))
		{
			// strip whitespace and remove end-of-line comments
			bool prev_isspace = true;
			int j = 0;
			for (int i=0;  i < 4096 && s[i] != 0;  ++i)
			{
				if (s[i] == '#')
				{
					// rest of line is comment
					break;
				}
				else if (isspace(s[i]))
				{
					if (!prev_isspace)
					{
						// insert one whitespace
						s[j++] = ' ';
					}
					prev_isspace = true;
				}
				else
				{
					s[j++] = s[i];
					prev_isspace = false;
				}
			}
			s[j] = 0;  // don't forget string termination

			++line_num;
			char voxet_path[4096];
			if (sscanf(s, "USE VOXET %s", voxet_path) == 1)
			{
				if (_voxet != 0L)
				{
					printf("%s (line %d) : Error - USE VOXET cannot appear more than once in file.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				else
				{
					_voxet = new Voxet(_log_level,voxet_path);
					if (_voxet->Get_Global_Coordinate_System() == 0L)
					{
						printf("%s (line %d) : Error - Voxet contains no global coordinate system information.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						continue;
					}
				}
			}
			char transpose[4096];
			if (!error && sscanf(s, "TRANSPOSE UVW = %s", transpose) == 1)
			{
				if (_voxet == 0L)
				{
					printf("%s (line %d): Error - TRANSPOSE UVW cannot appear before USE VOXET.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				else
				{
					_tolower(transpose);
					if (_voxet->Get_Global_Coordinate_System()->Set_Transpose(transpose))
					{
						if (_log_level > 3) printf("Transpose set to uvw -> %s\n",transpose);
						continue;
					}
					else
					{
						printf("%s (line %d) : Error - Set transpose to uvw -> %s failed.\n",parmfile_path,line_num);
						error = true;
						break;
					}
				}
			}
			char isostr[4096];
			if (!error && sscanf(s, "USE_ISOTROPIC_SPHERE_DURING_SOURCE_INJECTION %s", isostr) == 1)
			{
				_tolower(isostr);
				if (strcmp(isostr, "enabled") == 0) _use_isotropic_sphere_during_source_injection = true;
				if (_log_level >= 3) printf("Isotropic sphere will be used during source injection.\n");
			}
			char property[4096];
			char moniker[4096];
			double min=0.0, max=-1.0;
			if (!error && sscanf(s, "PROPERTY %s = %s %lf %lf", property, moniker, &min, &max) == 4)
			{
				if (_voxet == 0L)
				{
					printf("%s (line %d): Error - PROPERTY cannot appear before USE VOXET.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				else
				{
					int attr_idx = Get_Earth_Model_Attribute_Index(property);
					if (attr_idx < 0)
					{
						printf("%s (line %d): Error - Unknown property %s.\n",parmfile_path,line_num,property);
						printf("The following properties are allowed:\n");
						for (int i = 0;  i < _num_em_props;  ++i)
						{
							printf("  %s\n",Get_Earth_Model_Attribute_Moniker(i));
						}
						error = true;
						break;
					}
					else
					{
						Voxet_Property* prop = _voxet->Get_Property_By_Moniker(moniker);
						if (prop == 0L)
						{
							printf("%s (line %d): Error - Voxet file does not have property %s.\n",parmfile_path,line_num,moniker);
							printf("The voxet has the following properties:\n");
							for (int i = 0;  i < _voxet->Get_Number_Of_Properties();  ++i)
							{
								printf("  %s\n",_voxet->Get_Property_By_Index(i)->Get_Moniker());
							}
							error = true;
							break;
						}
						else
						{
							if (attr_idx == Attr_Idx_Q)
							{
								float tmp = 1.0f / max;
								max = 1.0f / min;
								min = tmp;
							}
							_props[attr_idx] = prop;
							if (min < max) _props[attr_idx]->Set_MinMax(min,max);
							continue;
						}
					}					
				}
			}
			double const_val;
			if (!error && sscanf(s, "PROPERTY %s = %lf", property, &const_val) == 2)
			{
				if (_voxet == 0L)
                                {
                                        printf("%s (line %d): Error - PROPERTY cannot appear before USE VOXET.\n",parmfile_path,line_num);
                                        error = true;
					break;
                                }
                                else
                                {
					int attr_idx = Get_Earth_Model_Attribute_Index(property);
					if (attr_idx < 0)
					{
						printf("%s (line %d): Error - Unknown property %s.\n",parmfile_path,line_num,property);
						printf("The following properties are allowed:\n");
						for (int i = 0;  i < _num_em_props;  ++i)
						{
							printf("  %s\n",Get_Earth_Model_Attribute_Moniker(i));
						}
						error = true;
						break;
					}
					else
					{
						_props[attr_idx] = 0L;
						_const_vals[attr_idx] = attr_idx == Attr_Idx_Q ? 1.0f / const_val : const_val;
						continue;
					}					
				}
			}
			double fq;
			if (!error && sscanf(s, "SET FQ = %lf", &fq) == 1)
			{
				if (fq <= 0.0)
				{
					printf("%s (line %d) : Error - FQ of %lf is not physical.\n",parmfile_path,line_num,fq);
					error = true;
					break;
				}
				else
				{	
					_fq = fq;
					if (_log_level > 3) printf("FQ set to %lf Hz.\n",_fq);
				}
			}
			char sub_origin[4096];
			if (!error && sscanf(s, "PROPAGATE_ORIGIN = %s", sub_origin) == 1)
			{
				if (_sub_x_set)
				{
					printf("%s (line %d): Error - PROPAGATE_ORIGIN must appear before PROPAGATE_X.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				if (_sub_y_set)
				{
					printf("%s (line %d): Error - PROPAGATE_ORIGIN must appear before PROPAGATE_Y.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				if (_sub_z_set)
				{
					printf("%s (line %d): Error - PROPAGATE_ORIGIN must appear before PROPAGATE_Z.\n",parmfile_path,line_num);
					error = true;
					break;
				}
				_tolower(sub_origin);
				if (strcmp(sub_origin, "source") == 0)
				{
					_sub_origin = 0;
					if (_log_level >= 3) printf("Sub volume is relative to source location.\n");
				}
				else if (strcmp(sub_origin, "volume") == 0)
				{
					_sub_origin = 1;
					if (_log_level >= 3) printf("Sub volume is relative to volume origin.\n");
				}
				else
				{
					printf("%s (line %d): Error - PROPAGATE_ORIGIN invalid origin string %s. Should be either Volume or Source.\n",parmfile_path,line_num,sub_origin);
					error = true;
					break;
				}
			}
			double sub_min, sub_max;
			char sub_unit[4096];
			if (!error && sscanf(s, "PROPAGATE_X = %lf %lf %s", &sub_min, &sub_max, sub_unit) == 3)
			{
				if (_voxet == 0L)
                                {
                                        printf("%s (line %d): Error - PROPAGATE_X cannot appear before USE VOXET.\n",parmfile_path,line_num);
                                        error = true;
                                        break;
                                }
                                else
                                {
					Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
					error = _Calculate_Sub_Volume("PROPAGATE_X",parmfile_path,line_num,gcs->Get_NX(),gcs->Get_DX(),sub_min,sub_max,sub_unit,_parm_sub_ix0,_parm_sub_ix1);
					if (error) break;
					_sub_x_set = true;
				}
			}
			if (!error && sscanf(s, "PROPAGATE_Y = %lf %lf %s", &sub_min, &sub_max, sub_unit) == 3)
			{
				if (_voxet == 0L)
                                {
                                        printf("%s (line %d): Error - PROPAGATE_Y cannot appear before USE VOXET.\n",parmfile_path,line_num);
                                        error = true;
                                        break;
                                }
                                else
                                {
					Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
					error = _Calculate_Sub_Volume("PROPAGATE_Y",parmfile_path,line_num,gcs->Get_NY(),gcs->Get_DY(),sub_min,sub_max,sub_unit,_parm_sub_iy0,_parm_sub_iy1);
					if (error) break;
					_sub_y_set = true;
				}
			}
			if (!error && sscanf(s, "PROPAGATE_Z = %lf %lf %s", &sub_min, &sub_max, sub_unit) == 3)
			{
				if (_voxet == 0L)
                                {
                                        printf("%s (line %d): Error - PROPAGATE_Z cannot appear before USE VOXET.\n",parmfile_path,line_num);
                                        error = true;
                                        break;
                                }
                                else
                                {
					Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
					error = _Calculate_Sub_Volume("PROPAGATE_Z",parmfile_path,line_num,gcs->Get_NZ(),gcs->Get_DZ(),sub_min,sub_max,sub_unit,_parm_sub_iz0,_parm_sub_iz1);
					if (error) break;
					_sub_z_set = true;
				}
			}
			double abc_size;
			char abc_unit[4096];
			char abc_flag[4096];
			if (!error)
			{
				int matched = sscanf(s, "NABC_SDX = %lf %s %s", &abc_size, abc_unit, abc_flag);
				if (matched == 2 || matched == 3)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - NABC_SDX cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
						error = _Calculate_ABC_Sponge("NABC_SDX",parmfile_path,line_num,abc_size,abc_unit,matched==3?abc_flag:0L,gcs->Get_NX(),gcs->Get_DX(),_parm_nabc_sdx,_parm_nabc_sdx_extend);
						if (error) break;
					}
				}
			}
			if (!error)
			{
				int matched = sscanf(s, "NABC_SDY = %lf %s %s", &abc_size, abc_unit, abc_flag);
				if (matched == 2 || matched == 3)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - NABC_SDY cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
						error = _Calculate_ABC_Sponge("NABC_SDY",parmfile_path,line_num,abc_size,abc_unit,matched==3?abc_flag:0L,gcs->Get_NY(),gcs->Get_DY(),_parm_nabc_sdy,_parm_nabc_sdy_extend);
						if (error) break;
					}
				}
			}
			if (!error)
			{
				int matched = sscanf(s, "NABC_TOP = %lf %s %s", &abc_size, abc_unit, abc_flag);
				if (matched == 2 || matched == 3)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - NABC_TOP cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
						error = _Calculate_ABC_Sponge("NABC_TOP",parmfile_path,line_num,abc_size,abc_unit,matched==3?abc_flag:0L,gcs->Get_NZ(),gcs->Get_DZ(),_parm_nabc_top,_parm_nabc_top_extend);
						if (error) break;
					}
				}
			}
			if (!error)
			{
				int matched = sscanf(s, "NABC_BOT = %lf %s %s", &abc_size, abc_unit, abc_flag);
				if (matched == 2 || matched == 3)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - NABC_BOT cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
						error = _Calculate_ABC_Sponge("NABC_BOT",parmfile_path,line_num,abc_size,abc_unit,matched==3?abc_flag:0L,gcs->Get_NZ(),gcs->Get_DZ(),_parm_nabc_bot,_parm_nabc_bot_extend);
						if (error) break;
					}
				}
			}
			if (!error)
			{
				char lower_Q_seafloor_flag_str[4096];
				int matched = sscanf(s, "LOWER_Q_ALONG_SEAFLOOR %s", lower_Q_seafloor_flag_str);
				if (matched == 1)
				{
					_tolower(lower_Q_seafloor_flag_str);
					if (strcmp(lower_Q_seafloor_flag_str, "enabled") == 0)
					{
						_lower_Q_seafloor_enabled = true;
						if (_log_level >= 3) printf("Q will be lowered to 10 along seafloor to attenuate Scholte waves.\n");
					}
				}
			}
			if (!error)
			{
				int souidx;
				double sou_x, sou_y, sou_z;
				char sub_unit[4096];
				int matched = sscanf(s, "SHOT %d SOURCE_LOCATION %lf %lf %lf %s", &souidx, &sou_x, &sou_y, &sou_z, sub_unit);
				if (matched == 5)
				{
					if (_voxet == 0L)
                                        {
                                                printf("%s (line %d): Error - SOURCE_LOCATION cannot appear before USE VOXET.\n",parmfile_path,line_num);
                                                error = true;
                                                break;
                                        }
                                        else
                                        {
						Elastic_Shot* shot = Get_Shot(souidx);
						if (shot != 0L)
						{
							printf("%s (line %d): Error - Multiple source locations for shot %d.\n",parmfile_path,line_num,souidx);
							error = true;
							break;
						}
						else
						{
							Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
							// convert source location to local coordinates
							_tolower(sub_unit);
							if (strcmp(sub_unit, "global") == 0)
							{
								double x, y, z;
								gcs->Convert_Global_To_Transposed_Fractional_Index(sou_x,sou_y,sou_z,x,y,z);
								shot = new Elastic_Shot(_log_level,this,souidx,x,y,z);
								Add_Shot(shot);
								printf("Shot %d :: Source location global=(%lf,%lf,%lf) index=(%lf,%lf,%lf)\n",
									shot->Get_Source_Index(),
									sou_x,sou_y,sou_z,
									x,y,z);
							}
							else if (strcmp(sub_unit, "local") == 0)
							{
								double x = sou_x / gcs->Get_DX();
								double y = sou_y / gcs->Get_DY();
								double z = sou_z / gcs->Get_DZ();
								shot = new Elastic_Shot(_log_level,this,souidx,x,y,z);
								Add_Shot(shot);
								printf("Shot %d :: Source location local=(%lf,%lf,%lf) index=(%lf,%lf,%lf)\n",
									shot->Get_Source_Index(),
									sou_x,sou_y,sou_z,
									x,y,z);
							}
							else if (strcmp(sub_unit, "index") == 0)
							{
								// no conversion
								shot = new Elastic_Shot(_log_level,this,souidx,sou_x,sou_y,sou_z);
								Add_Shot(shot);
								printf("Shot %d :: Source location index=(%lf,%lf,%lf)\n",
									shot->Get_Source_Index(),
									sou_x,sou_y,sou_z);
							}
							else
							{
								printf("%s (line %d): Error - SOURCE_LOCATION sub unit '%s' not supported.\n",sub_unit);
								error = true;
								break;
							}
						}
                                        }
				}
			}
			if (!error)
			{
				int souidx;
				char stype[4096];
				int matched = sscanf(s, "SHOT %d SOURCE_TYPE %s", &souidx, stype);
				if (matched == 2)
				{
					Elastic_Shot* shot = Get_Shot(souidx);
					if (shot == 0L)
					{
						printf("%s (line %d): Error - SOURCE_TYPE Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
						error = true;
						break;
					}
					else
					{
						_tolower(stype);
						if (strcmp(stype, "force") == 0)
						{
							shot->Set_Source_Type(Elastic_Shot::Source_Type_Force);
							if (_log_level > 3) printf("Shot %d :: SOURCE_TYPE set to force.\n",shot->Get_Source_Index());
						}
						else if(strcmp(stype, "velocity") == 0)
						{
							shot->Set_Source_Type(Elastic_Shot::Source_Type_Velocity);
							if (_log_level > 3) printf("Shot %d :: SOURCE_TYPE set to velocity.\n",shot->Get_Source_Index());
						}
						else if(strcmp(stype, "pressure") == 0)
						{
							shot->Set_Source_Type(Elastic_Shot::Source_Type_Pressure);
							if (_log_level > 3) printf("Shot %d :: SOURCE_TYPE set to pressure.\n",shot->Get_Source_Index());
						}
						else
						{
							printf("%s (line %d): Error - SOURCE_TYPE invalid source type '%s'.\n",parmfile_path,line_num,stype);
							error = true;
							break;
						}
					}
				}
			}
			if (!error)
			{
				int souidx;
				double ampl1, ampl2, ampl3;
				int matched = sscanf(s, "SHOT %d SOURCE_AMPLITUDE %lf %lf %lf", &souidx, &ampl1, &ampl2, &ampl3);
				if (matched == 2 || matched == 4)
				{
					Elastic_Shot* shot = Get_Shot(souidx);
                                        if (shot == 0L)
                                        {
                                                printf("%s (line %d): Error - SOURCE_AMPLITUDE Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
                                                error = true;
                                                break;
                                        }
                                        else
					{
						if (matched == 2)
						{
							if (shot->Get_Source_Type() != Elastic_Shot::Source_Type_Pressure)
							{
								printf("%s (line %d): Error - SOURCE_AMPLITUDE this source type (%s) requires 3 amplitudes.\n",parmfile_path,line_num,shot->Get_Source_Type_String());
								error = true;
								break;
							}
							else
							{
								shot->Set_Amplitudes(ampl1,0.0,0.0);
								if (_log_level > 3) printf("Shot %d :: SOURCE_AMPLITUDE set to %lf\n",shot->Get_Source_Index(),ampl1);
							}
						}
						else if (matched == 4)
						{
							if (shot->Get_Source_Type() == Elastic_Shot::Source_Type_Pressure)
							{
								printf("%s (line %d): Error - SOURCE_AMPLITUDE this source type (%s) requires 1 amplitude.\n",parmfile_path,line_num,shot->Get_Source_Type_String());
								error = true;
								break;
							}
							else
							{
								if (ampl1 == 0.0 && ampl2 == 0.0 && ampl3 == 0.0)
								{
									printf("%s (line %d): Error - SOURCE_AMPLITUDE at least one amplitude must be non zero.\n",parmfile_path,line_num);
									error = true;
									break;
								}
								else
								{
									shot->Set_Amplitudes(ampl1,ampl2,ampl3);
									if (_log_level > 3) printf("Shot %d :: SOURCE_AMPLITUDE set to %lf, %lf, %lf\n",shot->Get_Source_Index(),ampl1,ampl2,ampl3);
								}
							}
						}
					}
				}
			}
			if (!error)
			{
				int souidx;
				char wavetype[4096];
				double freq;
				int matched = sscanf(s, "SHOT %d SOURCE_WAVELET %s %lf", &souidx, wavetype, &freq);
				if (matched == 3)
				{
					Elastic_Shot* shot = Get_Shot(souidx);
                                        if (shot == 0L)
                                        {
                                                printf("%s (line %d): Error - SOURCE_WAVELET Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
                                                error = true;
                                                break;
                                        }
                                        else
                                        {
						_tolower(wavetype);
						error = shot->Use_Builtin_Source_Wavelet(wavetype, freq, parmfile_path, line_num);
						if (error) break;
						if (_log_level > 3) printf("Shot %d :: Using builtin %s wavelet with maximum frequency of %.2lfHz\n",shot->Get_Source_Index(),wavetype,freq);
					}
				}
			}
			if (!error)
			{
				int souidx;
				char wavelet_path[4096];
				double fmax;
				int filter_order;
				int matched = sscanf(s, "SHOT %d SOURCE_WAVELET FILE %s %lf %d", &souidx, wavelet_path, &fmax, &filter_order);
				if (matched == 4)
				{
					Elastic_Shot* shot = Get_Shot(souidx);
                                        if (shot == 0L)
                                        {
                                                printf("%s (line %d): Error - SOURCE_WAVELET FILE Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
                                                error = true;
                                                break;
                                        }
                                        else
                                        {
						error = shot->Read_Source_Wavelet_From_File(wavelet_path,fmax,filter_order);
                                                if (error) break;
                                                if (_log_level > 3) printf("Shot %d :: Source wavelet will be read from file %s and filtered to comply with F_max=%.2fHz.\n",shot->Get_Source_Index(),wavelet_path,fmax);
                                        }
				}
			}
			if (!error)
			{
				int souidx, fileidx;
				char base_filename[4096];
				double sample_rate;
				double tshift;
				double reclen;
				char field1[256];
				char field2[256];
				char field3[256];
				char field4[256];
				int matched = sscanf(s, "SHOT %d SEGY_FILE %d FILE %s %lf %lf %lf %s %s %s %s", &souidx, &fileidx, base_filename, &sample_rate, &tshift, &reclen, field1, field2, field3, field4);
				if (matched >= 7 && matched <= 10)
				{
					Elastic_Shot* shot = Get_Shot(souidx);
                                        if (shot == 0L)
                                        {
						printf("%s (line %d): Error - SEGY_FILE Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
						error = true;
						break;
					}
					else
					{
						if (shot->Get_SEGY_File(fileidx) != 0L)
						{
							printf("%s (line %d): Error - SEGY_FILE duplicate definiton of file with file index %d.\n",parmfile_path,line_num,fileidx);
							error = true;
							break;
						}
						else
						{
							bool do_P=false, do_Vx=false, do_Vy=false, do_Vz=false;
							_tolower(field1);
							if (strcmp(field1, "p") == 0) do_P = true;
							else if (strcmp(field1, "vx") == 0) do_Vx = true;
							else if (strcmp(field1, "vy") == 0) do_Vy = true;
							else if (strcmp(field1, "vz") == 0) do_Vz = true;
							if (matched > 7)
							{
								_tolower(field2);
								if (strcmp(field2, "p") == 0) do_P = true;
								else if (strcmp(field2, "vx") == 0) do_Vx = true;
								else if (strcmp(field2, "vy") == 0) do_Vy = true;
								else if (strcmp(field2, "vz") == 0) do_Vz = true;
								if (matched > 8)
								{
									_tolower(field3);
									if (strcmp(field3, "p") == 0) do_P = true;
									else if (strcmp(field3, "vx") == 0) do_Vx = true;
									else if (strcmp(field3, "vy") == 0) do_Vy = true;
									else if (strcmp(field3, "vz") == 0) do_Vz = true;
									if (matched > 9)
									{
										_tolower(field4);
										if (strcmp(field4, "p") == 0) do_P = true;
										else if (strcmp(field4, "vx") == 0) do_Vx = true;
										else if (strcmp(field4, "vy") == 0) do_Vy = true;
										else if (strcmp(field4, "vz") == 0) do_Vz = true;
									}
								}
							}
							Elastic_SEGY_File* segy_file = new Elastic_SEGY_File(fileidx,base_filename,sample_rate,tshift,reclen,do_P,do_Vx,do_Vy,do_Vz);
							shot->Add_SEGY_File(segy_file);
							if (_log_level >= 3)
							{
								printf("Added SEGY FILE with idx %d to shot with source idx %d.\n",fileidx,souidx);
								printf("...sample rate set to %lfs\n",sample_rate);
								printf("...record length set to %lfs\n",reclen);
								printf("...time shift set to %lfs\n",tshift);
								printf("...outputting wavefields %s %s %s %s\n",do_P?"P":"",do_Vx?"Vx":"",do_Vy?"Vy":"",do_Vz?"Vz":"");
							}
						}
					}
				}
			}
			if (!error)
			{
				int souidx, fileidx;
				char gather_type_str[4096];
				int matched = sscanf(s, "SHOT %d SEGY_FILE %d GATHER_TYPE %s", &souidx, &fileidx, gather_type_str);
				if (matched == 3)
				{
					_tolower(gather_type_str);
					Elastic_Shot* shot = Get_Shot(souidx);
					if (shot == 0L)
					{
						printf("%s (line %d): Error - SEGY_FILE GATHER_TYPE shot with source index %d not found.\n",parmfile_path,line_num,souidx);
						error = true;
						break;
					}
					else
					{
						Elastic_SEGY_File* segy_file = shot->Get_SEGY_File(fileidx);
						if (segy_file == 0L)
						{
							printf("%s (line %d): Error - SEGY_FILE GATHER_TYPE file with index %d not found.\n",parmfile_path,line_num,fileidx);
							error = true;
							break;
						}
						else
						{
							if (strcmp(gather_type_str,"common_receiver_gather") == 0)
							{
								segy_file->Set_Gather_Type(Common_Receiver_Gather);
								if (_log_level >= 3) printf("Gather type set to %s for segy file %d in shot %d.\n",ToString_Elastic_Gather_Type_t(segy_file->Get_Gather_Type()),souidx,fileidx);
							}
						}
					}
				}
			}
			if (!error)
			{
				int souidx, fileidx, rangeidx;
				double start, end, interval;
				char unit[4096];
				int matched = sscanf(s, "SHOT %d SEGY_FILE %d RECEIVER_LOCATIONS %d RANGE_X %lf %lf %lf %s", &souidx, &fileidx, &rangeidx, &start, &end, &interval, unit);
				if (matched == 7)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_X cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Elastic_Shot* shot = Get_Shot(souidx);
						if (shot == 0L)
						{
							printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_X Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
							error = true;
							break;
						}
						else
						{
							Elastic_SEGY_File* segy_file = shot->Get_SEGY_File(fileidx);
							if (segy_file == 0L)
							{
								printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_X SEGY file with index %d not found.\n",parmfile_path,line_num,fileidx);
								error = true;
								break;
							}
							else
							{
								Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
								_tolower(unit);
								if (strcmp(unit, "local") == 0)
								{
									start = start / gcs->Get_DX();
									end = end / gcs->Get_DX();
									interval = interval / gcs->Get_DX();
									segy_file->Add_Receiver_Range_X(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added X=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else if (strcmp(unit, "index") == 0)
								{
									// no conversion
									segy_file->Add_Receiver_Range_X(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added X=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else
								{
									printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_X unit %s not supported.\n",parmfile_path,line_num,unit);
									error = true;
									break;
								}
							}
						}
					}
				}
			}
			if (!error)
			{
				int souidx, fileidx, rangeidx;
				double start, end, interval;
				char unit[4096];
				int matched = sscanf(s, "SHOT %d SEGY_FILE %d RECEIVER_LOCATIONS %d RANGE_Y %lf %lf %lf %s", &souidx, &fileidx, &rangeidx, &start, &end, &interval, unit);
				if (matched == 7)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Y cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Elastic_Shot* shot = Get_Shot(souidx);
						if (shot == 0L)
						{
							printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Y Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
							error = true;
							break;
						}
						else
						{
							Elastic_SEGY_File* segy_file = shot->Get_SEGY_File(fileidx);
							if (segy_file == 0L)
							{
								printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Y SEGY file with index %d not found.\n",parmfile_path,line_num,fileidx);
								error = true;
								break;
							}
							else
							{
								Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
								_tolower(unit);
								if (strcmp(unit, "local") == 0)
								{
									start = start / gcs->Get_DY();
									end = end / gcs->Get_DY();
									interval = interval / gcs->Get_DY();
									segy_file->Add_Receiver_Range_Y(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added Y=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else if (strcmp(unit, "index") == 0)
								{
									// no conversion
									segy_file->Add_Receiver_Range_Y(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added Y=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else
								{
									printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Y unit %s not supported.\n",parmfile_path,line_num,unit);
									error = true;
									break;
								}
							}
						}
					}
				}
			}
			if (!error)
			{
				int souidx, fileidx, rangeidx;
				double start, end, interval;
				char unit[4096];
				int matched = sscanf(s, "SHOT %d SEGY_FILE %d RECEIVER_LOCATIONS %d RANGE_Z %lf %lf %lf %s", &souidx, &fileidx, &rangeidx, &start, &end, &interval, unit);
				if (matched == 7)
				{
					if (_voxet == 0L)
					{
						printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Z cannot appear before USE VOXET.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						Elastic_Shot* shot = Get_Shot(souidx);
						if (shot == 0L)
						{
							printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Z Shot with source index %d not found.\n",parmfile_path,line_num,souidx);
							error = true;
							break;
						}
						else
						{
							Elastic_SEGY_File* segy_file = shot->Get_SEGY_File(fileidx);
							if (segy_file == 0L)
							{
								printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Z SEGY file with index %d not found.\n",parmfile_path,line_num,fileidx);
								error = true;
								break;
							}
							else
							{
								Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
								_tolower(unit);
								if (strcmp(unit, "local") == 0)
								{
									start = start / gcs->Get_DZ();
									end = end / gcs->Get_DZ();
									interval = interval / gcs->Get_DZ();
									segy_file->Add_Receiver_Range_Z(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added Z=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else if (strcmp(unit, "index") == 0)
								{
									// no conversion
									segy_file->Add_Receiver_Range_Z(rangeidx,start,end,interval);
									if (_log_level >= 3) printf("Added Z=%lf,%lf,%lf to range with index %d to SEGY file with index %d.\n",start,end,interval,rangeidx,fileidx);
								}
								else
								{
									printf("%s (line %d): Error - SEGY_FILE RECEIVER_LOCATIONS RANGE_Z unit %s not supported.\n",parmfile_path,line_num,unit);
									error = true;
									break;
								}
							}
						}
					}
				}
			}
			if (!error)
			{
				float cf;
				int matched = sscanf(s, "COURANT_FACTOR = %f", &cf);
				if (matched == 1)
				{
					if (cf <= 0.0f)
					{
						printf("%s (line %d): Error - COURANT_FACTOR must be positive.\n",parmfile_path,line_num);
						error = true;
						break;
					}
					else
					{
						_Courant_Factor = cf;
						if (_log_level >= 3) printf("Courant factor set to %f.\n",_Courant_Factor);
					}
				}
			}
			char flag[4096];
			if (sscanf(s, "FREESURFACE = %s", flag) == 1)
			{
				_tolower(flag);
				_freesurface_enabled = strcmp(flag, "enabled") == 0 ? true : false;
				if (_log_level > 3) printf("FREESURFACE is %s.\n", _freesurface_enabled?"enabled":"disabled");
			}
			if (sscanf(s, "SOURCE_GHOST = %s", flag) == 1)
			{
				_tolower(flag);
				_source_ghost_enabled = strcmp(flag, "enabled") == 0 ? true : false;
				if (_log_level > 3) printf("SOURCE_GHOST is %s.\n", _source_ghost_enabled?"enabled":"disabled");
			}
			if (sscanf(s, "RECEIVER_GHOST = %s", flag) == 1)
			{
				_tolower(flag);
				_receiver_ghost_enabled = strcmp(flag, "enabled") == 0 ? true : false;
				if (_log_level > 3) printf("RECEIVER_GHOST is %s.\n", _receiver_ghost_enabled?"enabled":"disabled");
			}
			char GPU_Devices[4096];
			if (sscanf(s, "GPU_DEVICES = %s", GPU_Devices) == 1)
			{
				if (_GPU_Devices != 0L) delete [] _GPU_Devices;
				_GPU_Devices = new int[1024];
				_num_GPU_Devices = 0;
				char* tok = strtok(GPU_Devices, ",");
				while (tok != 0L)
				{
					_GPU_Devices[_num_GPU_Devices++] = atoi(tok);
					tok = strtok(0L, ",");
				}
				if (_log_level > 3)
				{
					printf("GPU_DEVICES = ");
					for (int i = 0;  i < _num_GPU_Devices;  ++i)
					{
						if (i == 0)
						{
							printf("%d", _GPU_Devices[i]);
						}
						else
						{
							printf(", %d",_GPU_Devices[i]);
						}
					}
					printf("\n");
				}
			}
			int GPU_Pipes;
			if (sscanf(s, "GPU_PIPES = %d", &GPU_Pipes) == 1)
			{
				_GPU_Pipes = GPU_Pipes;
				if (_log_level > 3) printf("GPU_PIPES = %d\n",_GPU_Pipes);
			}
			int Steps_Per_GPU;
			if (sscanf(s, "STEPS_PER_GPU = %d", &Steps_Per_GPU) == 1)
			{
				_Steps_Per_GPU = Steps_Per_GPU;
				if (_log_level > 3) printf("STEPS_PER_GPU = %d\n", _Steps_Per_GPU);
			}
		}
		fclose(fp);
	}
	if (!error)
	{
		if (_voxet == 0L)
		{
			printf("%s : Error - USE VOXET line was not found.\n",parmfile_path);
			error = true;
		}
		else
		{
			// verify size and existence of property binary files
			Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
			size_t expected_file_size = (size_t)4 * (size_t)gcs->Get_NU() * (size_t)gcs->Get_NV() * (size_t)gcs->Get_NW();
			for (int i = 0;  i < _num_em_props;  ++i)
			{
				error = error || _Check_Property(Get_Earth_Model_Attribute_Moniker(i),_props[i],_const_vals[i],expected_file_size);
			}
			_Is_Valid = !error;
			if (_Is_Valid)
			{
				Compute_Subvolume();
				for (int i = 0;  i < _num_em_props;  ++i)
				{
					Voxet_Property* prop = _props[i];
					if (prop == 0L)
					{
						_pck_min[i] = _const_vals[i];
						_pck_max[i] = _const_vals[i];
						_pck_iso[i] = _const_vals[i];
					}
					else
					{
						if (!prop->Has_MinMax())
						{
							prop->Get_MinMax_From_File();
							if (i == Attr_Idx_Q) prop->Set_MinMax(1.0f/prop->Get_Max(), 1.0f/prop->Get_Min()); // for Q we pack reciprocal of Q
						}
						if (i == Attr_Idx_Q && _lower_Q_seafloor_enabled && prop->Get_Max() < 0.1f)
						{
							prop->Set_MinMax(prop->Get_Min(), 0.1f);
							if (_log_level >= 3) printf("Minimum Q lowered to %f\n",1.0f/prop->Get_Max());
						}
						_pck_min[i] = prop->Get_Min();
						_pck_max[i] = prop->Get_Max();
						if (i == Attr_Idx_Vp || i == Attr_Idx_Density || i == Attr_Idx_Q)
							_pck_iso[i] = prop->Get_Min();
						else
							_pck_iso[i] = 0.0f;
					}
					if (_pck_min[i] == _pck_max[i]) _pck_max[i] = _pck_min[i] + fabs(0.1f * _pck_min[i]);
					_pck_range[i] = _pck_max[i] - _pck_min[i];
					//printf("%s - Min=%f, Range=%f\n",_pck_moniker[i],_pck_min[i],_pck_range[i]);
				}
			}
			if (_Is_Valid)
			{
				// verify receiver locations for all shots
				for (int i = 0;  i < _num_shots;  ++i)
				{
					Elastic_Shot* shot = _shots[i];
						
				}
			}
			if (_log_level > 2) printf("Parameter file appears to be %s.\n",_Is_Valid?"valid":"invalid");
		}
	}
}

void Elastic_Modeling_Job::Compute_Subvolume()
{
	// copy sub volume parameters from parameter file
	_sub_ix0 = _parm_sub_ix0;
	_sub_ix1 = _parm_sub_ix1;
	_sub_iy0 = _parm_sub_iy0;
	_sub_iy1 = _parm_sub_iy1;
	_sub_iz0 = _parm_sub_iz0;
	_sub_iz1 = _parm_sub_iz1;
	_nabc_sdx = _parm_nabc_sdx;
	_nabc_sdy = _parm_nabc_sdy;
	_nabc_top = _parm_nabc_top;
	_nabc_bot = _parm_nabc_bot;
	_nabc_sdx_extend = _parm_nabc_sdx_extend;
	_nabc_sdy_extend = _parm_nabc_sdy_extend;
	_nabc_top_extend = _parm_nabc_top_extend;
	_nabc_bot_extend = _parm_nabc_bot_extend;

	// determine sub volume dimensions
	int ghost_padding = 0;
	if (_freesurface_enabled)
	{
		// force to zero if freesurface is enabled.
		_nabc_top = 0;
		_nabc_top_extend = false;
	}
	else
	{
		// force top sponge to extend
		_nabc_top_extend = true;
		for (int iShot = 0;  iShot < Get_Number_Of_Shots();  ++iShot)
		{
			Elastic_Shot* shot = Get_Shot_By_Index(iShot);
			if (_source_ghost_enabled)
			{
				int zs = (int)lrintf(shot->Get_Propagation_Source_Z()) + 1;
				if (zs > ghost_padding) ghost_padding = zs;
			}
			if (_receiver_ghost_enabled)
			{
				int zr = (int)lrintf(shot->Find_Deepest_Receiver()) + 1;
				if (zr > ghost_padding) ghost_padding = zr;
			}
		}
	}
	if (Subvolume_Is_Relative_To_Source())	
	{
		// find smallest bounding box that includes all source locations.
		int src_min_x=1, src_max_x=0, src_min_y=1, src_max_y=0;
		for (int iShot = 0;  iShot < Get_Number_Of_Shots();  ++iShot)
		{
			Elastic_Shot* shot = Get_Shot_By_Index(iShot);
			int src_x0 = (int)floor(shot->Get_Source_X());
			int src_x1 = (int)ceil(shot->Get_Source_X());
			int src_y0 = (int)floor(shot->Get_Source_Y());
			int src_y1 = (int)ceil(shot->Get_Source_Y());
			if (src_min_x > src_max_x)
			{
				src_min_x = src_x0;
				src_max_x = src_x1;
			}
			else
			{
				if (src_x0 < src_min_x) src_min_x = src_x0;
				if (src_x1 > src_max_x) src_max_x = src_x1;
			}
			if (src_min_y > src_max_y)
			{
				src_min_y = src_y0;
				src_max_y = src_y1;
			}
			else
			{
				if (src_y0 < src_min_y) src_min_y = src_y0;
				if (src_y1 > src_max_y) src_max_y = src_y1;
			}
		}
		_sub_ix0 += src_min_x;
		_sub_ix1 += src_max_x;
		_sub_iy0 += src_min_y;
		_sub_iy1 += src_max_y;
	}
	// clip sub volume
	Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();
	if (_sub_ix0 < 0) _sub_ix0 = 0;
	if (_sub_ix1 >= gcs->Get_NX()) _sub_ix1 = gcs->Get_NX() - 1;
	if (_sub_iy0 < 0) _sub_iy0 = 0;
	if (_sub_iy1 >= gcs->Get_NY()) _sub_iy1 = gcs->Get_NY() - 1;
	if (_log_level > 3)
	{
		printf("X : Sub volume is [%d,%d]\n",_sub_ix0,_sub_ix1);
		printf("Y : Sub volume is [%d,%d]\n",_sub_iy0,_sub_iy1);
		printf("Z : Sub volume is [%d,%d]\n",_sub_iz0,_sub_iz1);
	}
	// X
	_prop_nx = _sub_ix1 - _sub_ix0 + 1;
	_prop_x0 = _sub_ix0;
	if (_nabc_sdx_extend)
	{
		_prop_nx += 2 * _nabc_sdx;
		_prop_x0 -= _nabc_sdx;
	}
	_prop_nx = ((_prop_nx + 3) >> 2) << 2;  // make prop_nx a multiple of 4.
	// expand sub-volume if extend is desired
	_sub_ix0 = _prop_x0 > 0 ? _prop_x0 : 0;
	int prop_x1 = _prop_x0 + _prop_nx - 1;
	_sub_ix1 = prop_x1 < gcs->Get_NX() ? prop_x1 : gcs->Get_NX() - 1;
	// Y
	_prop_ny = _sub_iy1 - _sub_iy0 + 1;
	_prop_y0 = _sub_iy0;
	if (_nabc_sdy_extend)
	{
		_prop_ny += 2 * _nabc_sdy;
		_prop_y0 -= _nabc_sdy;
	}
	_prop_ny = ((_prop_ny + 7) >> 3) << 3;  // make prop ny a multiple of 8
	// expand sub-volume if extend is desired
	_sub_iy0 = _prop_y0 > 0 ? _prop_y0 : 0;
	int prop_y1 = _prop_y0 + _prop_ny - 1;
	_sub_iy1 = prop_y1 < gcs->Get_NY() ? prop_y1 : gcs->Get_NY() - 1;
	// Z
	_prop_nz = _sub_iz1 - _sub_iz0 + 1;
	_prop_z0 = _sub_iz0;
	if (_nabc_top_extend)
	{
		_prop_nz += _nabc_top + ghost_padding;
		_prop_z0 -= (_nabc_top + ghost_padding);
	}
	if (_nabc_bot_extend)
	{
		_prop_nz += _nabc_bot;
	}
	_prop_nz = ((_prop_nz + 7) >> 3) << 3;  // make prop nz a multiple of 8
	// expand sub-volume if extend is desired
	_sub_iz0 = _prop_z0 > 0 ? _prop_z0 : 0;
	int prop_z1 = _prop_z0 + _prop_nz - 1;
	_sub_iz1 = prop_z1 < gcs->Get_NZ() ? prop_z1 : gcs->Get_NZ() - 1;
	if (_log_level > 3)
	{
		printf("Propagation volume X = [%d,%d].\n",_prop_x0,_prop_x0+_prop_nx-1);
		printf("Propagation volume Y = [%d,%d].\n",_prop_y0,_prop_y0+_prop_ny-1);
		printf("Propagation volume Z = [%d,%d].\n",_prop_z0,_prop_z0+_prop_nz-1);

		printf("X : Sub volume is [%d,%d]\n",_sub_ix0,_sub_ix1);
		printf("Y : Sub volume is [%d,%d]\n",_sub_iy0,_sub_iy1);
		printf("Z : Sub volume is [%d,%d]\n",_sub_iz0,_sub_iz1);
	}
}

void Elastic_Modeling_Job::Add_Shot(Elastic_Shot* shot)
{
	if (_shots == 0L)
	{
		_shots = new Elastic_Shot*[1];
		_shots[0] = shot;
		_num_shots = 1;
	}
	else
	{
		Elastic_Shot** new_shots = new Elastic_Shot*[_num_shots+1];
		for (int i = 0;  i < _num_shots;  ++i) new_shots[i] = _shots[i];
		new_shots[_num_shots] = shot;
		delete [] _shots;
		_shots = new_shots;
		++_num_shots;
	}
}

Elastic_Shot* Elastic_Modeling_Job::Get_Shot(int souidx)
{
	for (int i = 0;  i < _num_shots;  ++i)
	{
		if (_shots[i]->Get_Source_Index() == souidx)
		{
			return _shots[i];
		}
	}
	return 0L;
}

Elastic_Shot* Elastic_Modeling_Job::Get_Shot_By_Index(int idx)
{
	if (idx >= 0 && idx < _num_shots)
	{
		return _shots[idx];
	}
	return 0L;
}

int Elastic_Modeling_Job::Get_Propagation_NX()
{
	return _prop_nx;
}

int Elastic_Modeling_Job::Get_Propagation_NY()
{
	return _prop_ny;
}

int Elastic_Modeling_Job::Get_Propagation_NZ()
{
	return _prop_nz;
}

int Elastic_Modeling_Job::Get_Propagation_X0()
{
	return _prop_x0;
}

int Elastic_Modeling_Job::Get_Propagation_Y0()
{
	return _prop_y0;
}

int Elastic_Modeling_Job::Get_Propagation_Z0()
{
	return _prop_z0;
}

double Elastic_Modeling_Job::Get_DX()
{
	return _voxet->Get_Global_Coordinate_System()->Get_DX();
}

double Elastic_Modeling_Job::Get_DY()
{
	return _voxet->Get_Global_Coordinate_System()->Get_DY();
}

double Elastic_Modeling_Job::Get_DZ()
{
	return _voxet->Get_Global_Coordinate_System()->Get_DZ();
}

bool Elastic_Modeling_Job::Freesurface_Enabled()
{
	return _freesurface_enabled;
}

bool Elastic_Modeling_Job::Source_Ghost_Enabled()
{
	return _source_ghost_enabled;
}

bool Elastic_Modeling_Job::Receiver_Ghost_Enabled()
{
	return _receiver_ghost_enabled;
}

float Elastic_Modeling_Job::Get_Vpvert_Avg_Top()
{
	return _vpvert_avgtop;
}

float Elastic_Modeling_Job::Get_Vpvert_Avg_Bot()
{
	return _vpvert_avgbot;
}

int Elastic_Modeling_Job::Get_NABC_SDX()
{
	return _nabc_sdx;
}

int Elastic_Modeling_Job::Get_NABC_SDY()
{
	return _nabc_sdy;
}

int Elastic_Modeling_Job::Get_NABC_TOP()
{
	return _nabc_top;
}

int Elastic_Modeling_Job::Get_NABC_BOT()
{
	return _nabc_bot;
}

bool Elastic_Modeling_Job::Get_NABC_SDX_Extend()
{
	return _nabc_sdx_extend;
}

bool Elastic_Modeling_Job::Get_NABC_SDY_Extend()
{
	return _nabc_sdy_extend;
}

bool Elastic_Modeling_Job::Get_NABC_TOP_Extend()
{
	return _nabc_top_extend;
}

bool Elastic_Modeling_Job::Get_NABC_BOT_Extend()
{
	return _nabc_bot_extend;
}

void Elastic_Modeling_Job::Write_XZ_Slice(const char* path, int wf_type, int iy)
{
	FILE* fp = fopen(path, "w");
	if (fp != 0L)
	{
		for (int iz = 0;  iz < _prop_nz;  ++iz)
		{
			for (int ix = 0;  ix < _prop_nx;  ++ix)
			{
				fprintf(fp,"%d %d %e\n",ix,iz,_propagator->Get_Receiver_Value(wf_type, ix, iy, iz));
			}
			fprintf(fp,"\n");
		}
		char* wf;
		switch (wf_type)
		{
		case 0: // Vx
			wf = "Vx";
			break;
		case 1: // Vy
			wf = "Vy";
			break;
		case 2: // Vz
			wf = "Vz";
			break;
		case 3: // P
			wf = "P";
			break;
		case 6: // Txx
			wf = "Txx";
			break;
		case 7: // Tyy
			wf = "Tyy";
			break;
		case 8: // Tzz
			wf = "Tzz";
			break;
		case 9: // Txy
			wf = "Txy";
			break;
		case 10: // Txz
			wf = "Txz";
			break;
		case 11: // Tyz
			wf = "Tyz";
			break;
		default:
			wf = "??";
			break;
		}
		printf("X-Z Slice for %s wavefield written to %s.\n",wf,path);
		fclose(fp);
	}
	else
	{
		char buf[4096];
		sprintf(buf,"Write_XZ_Slice :: %s - ",path);
		perror(buf);
	}
}

void Elastic_Modeling_Job::Write_XY_Slice(const char* path, int wf_type, int iz)
{
	FILE* fp = fopen(path, "w");
	if (fp != 0L)
	{
		for (int iy = 0;  iy < _prop_ny;  ++iy)
		{
			for (int ix = 0;  ix < _prop_nx;  ++ix)
			{
				fprintf(fp,"%d %d %e\n",ix,iy,_propagator->Get_Receiver_Value(wf_type, ix, iy, iz));
			}
			fprintf(fp,"\n");
		}
		char* wf;
		switch (wf_type)
		{
		case 0: // Vx
			wf = "Vx";
			break;
		case 1: // Vy
			wf = "Vy";
			break;
		case 2: // Vz
			wf = "Vz";
			break;
		case 3: // P
			wf = "P";
			break;
		case 6: // Txx
			wf = "Txx";
			break;
		case 7: // Tyy
			wf = "Tyy";
			break;
		case 8: // Tzz
			wf = "Tzz";
			break;
		case 9: // Txy
			wf = "Txy";
			break;
		case 10: // Txz
			wf = "Txz";
			break;
		case 11: // Tyz
			wf = "Tyz";
			break;
		default:
			wf = "??";
			break;
		}
		printf("X-Y Slice for %s wavefield written to %s.\n",wf,path);
		fclose(fp);
	}
	else
	{
		char buf[4096];
		sprintf(buf,"Write_XY_Slice :: %s - ",path);
		perror(buf);
	}
}

void Elastic_Modeling_Job::_Pack_Earth_Model_Attribute(unsigned int& word, int attr_idx, float val)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
        {
		int ival = (int)round( (float)_pck_mask[attr_idx] * (val - _pck_min[attr_idx]) / _pck_range[attr_idx] );
		if (ival < 0) ival = 0;
		if (ival > _pck_mask[attr_idx]) ival = _pck_mask[attr_idx];
		word |= (((unsigned int)ival & _pck_mask[attr_idx]) << _pck_shft[attr_idx]);
		//printf("%s : val=%f, ival=%d (_pck_min=%f, _pck_range=%f, _pck_mask=%d)\n",Get_Earth_Model_Attribute_Moniker(attr_idx),val,ival,_pck_min[attr_idx],_pck_range[attr_idx],_pck_mask[attr_idx]);
	}
}

float Elastic_Modeling_Job::_Unpack_Earth_Model_Attribute(unsigned int word, int attr_idx)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		float norm_val = (float)((word >> _pck_shft[attr_idx]) & _pck_mask[attr_idx]) / (float)_pck_mask[attr_idx];
		//printf("%s - word = %d, normval = %f\n",Get_Earth_Model_Attribute_Moniker(attr_idx),word,norm_val);
		float unpack_val = _pck_min[attr_idx] + norm_val * _pck_range[attr_idx];
		return unpack_val;
	}
	else
	{
		printf("Elastic_Modeling_Job::_Unpack_Earth_Model_Attribute - Error, unknown attr_idx = %d\n",attr_idx);
		exit(0);
	}
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute(int attr_idx, int ix, int iy, int iz, bool& error)
{
	if (_propagator != 0L && attr_idx >= 0 && attr_idx < _num_em_props)
	{
		int widx = _pck_widx[attr_idx];
		unsigned int word = _propagator->_Get_Earth_Model_Word(widx,ix,iy,iz);
		error = false;
		return _Unpack_Earth_Model_Attribute(word,attr_idx);
	}
	else
	{
		error = true;
		return 0.0f;
	}
}

void Elastic_Modeling_Job::Set_Earth_Model_Attribute(int attr_idx, int ix, int iy, int iz, float new_value, bool& error)
{
	if (_propagator != 0L && attr_idx >= 0 && attr_idx < _num_em_props)
        {
                int widx = _pck_widx[attr_idx];
		unsigned int word = _propagator->_Get_Earth_Model_Word(widx,ix,iy,iz);
		_Pack_Earth_Model_Attribute(word,attr_idx,new_value);
		_propagator->_Set_Earth_Model_Word(widx,ix,iy,iz,word);
		error = false;
	}
	else
	{
		error = true;
	}	
}

void Elastic_Modeling_Job::Lower_Q_Seafloor()
{
	float Q_min_val = 10.0f;
	for (int ix = 0;  ix < _prop_nx;  ++ix)
	{
		for (int iy = 0;  iy < _prop_ny;  ++iy)
		{
			// locate sea floor
			bool error;
			int iz = -1;
			for (; iz < (_prop_nz-1) && Get_Earth_Model_Attribute(Attr_Idx_Density,ix,iy,iz+1,error) < 1.1f && !error;  ++iz);
			if (iz < (_prop_nz-3))
			{
				// set Q at sea floor plus two cells (vertically) to 10.
				// note that earth model is compressed at this point, so there is additional code that ensures min(Q) <= 10 in parser
				// if Q attenuation along sea floor is enabled.
				for (int my_iz = iz;  my_iz < iz+3;  ++my_iz)
				{
					float Q_val = 1.0f / Get_Earth_Model_Attribute(Attr_Idx_Q,ix,iy,my_iz);
					if (Q_val > Q_min_val) Set_Earth_Model_Attribute(Attr_Idx_Q,ix,iy,my_iz,1.0f/Q_min_val,error);
				}
			}
		}
	}
}

void Elastic_Modeling_Job::Write_Earth_Model_Attribute_XZ_Slice(const char* path, int attr_idx, int iy)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		char name[4096];
		sprintf(name, "%s_%s", path, Get_Earth_Model_Attribute_Moniker(attr_idx));
		FILE* fp = fopen(name, "w");
		if (fp != 0L)
		{
			//printf("Write_XZ :: x=[%d,%d], z=[%d,%d]\n",0,_prop_nx-1,0,_prop_nz-1);
			for (int iz = 0;  iz < _prop_nz;  ++iz)
			{
				for (int ix = 0;  ix < _prop_nx;  ++ix)
				{
					fprintf(fp,"%d %d %e\n",ix,iz,Get_Earth_Model_Attribute(attr_idx,ix,iy,iz));
				}
				fprintf(fp,"\n");
			}
			printf("X-Z Slice for earth model attribute %s written to %s.\n",Get_Earth_Model_Attribute_Moniker(attr_idx),name);
			fclose(fp);
		}
	}
	else
	{
		printf("Elastic_Modeling_Job::Write_Earth_Model_Attribute_XZ_Slice - Error, invalid attribute index %d\n",attr_idx);
		exit(0);
	}
}

void Elastic_Modeling_Job::Write_Earth_Model_XZ_Slice(const char* path, int iy)
{
	for (int attr_idx = 0;  attr_idx < _num_em_props;  ++attr_idx)
	{
		Write_Earth_Model_Attribute_XZ_Slice(path, attr_idx, iy);
	}
}

void Elastic_Modeling_Job::Write_Earth_Model_Attribute_XY_Slice(const char* path, int attr_idx, int iz)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		char name[4096];
		sprintf(name, "%s_%s", path, Get_Earth_Model_Attribute_Moniker(attr_idx));
		FILE* fp = fopen(name, "w");
		if (fp != 0L)
		{
			//printf("Write_XZ :: x=[%d,%d], z=[%d,%d]\n",0,_prop_nx-1,0,_prop_nz-1);
			for (int iy = 0;  iy < _prop_ny;  ++iy)
			{
				for (int ix = 0;  ix < _prop_nx;  ++ix)
				{
					fprintf(fp,"%d %d %e\n",ix,iy,Get_Earth_Model_Attribute(attr_idx,ix,iy,iz));
				}
				fprintf(fp,"\n");
			}
			printf("X-Y Slice for earth model attribute %s written to %s.\n",Get_Earth_Model_Attribute_Moniker(attr_idx),name);
			fclose(fp);
		}
	}
	else
	{
		printf("Elastic_Modeling_Job::Write_Earth_Model_Attribute_XY_Slice - Error, invalid attribute index %d\n",attr_idx);
		exit(0);
	}
}

void Elastic_Modeling_Job::Write_Earth_Model_XY_Slice(const char* path, int iz)
{
	for (int attr_idx = 0;  attr_idx < _num_em_props;  ++attr_idx)
	{
		Write_Earth_Model_Attribute_XY_Slice(path, attr_idx, iz);
	}
}

int Elastic_Modeling_Job::Get_Number_Of_Earth_Model_Attributes()
{
	return _num_em_props;
}

int Elastic_Modeling_Job::Get_Earth_Model_Attribute_Index(const char* moniker)
{
	for (int i = 0;  i < _num_em_props;  ++i)
	{
		if (strcmp(moniker, _pck_moniker[i]) == 0)
		{
			return i;
		}
	}
	return -1;
}

const char* Elastic_Modeling_Job::Get_Earth_Model_Attribute_Moniker(int attr_idx)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		return _pck_moniker[attr_idx];
	}
	else
	{
		return 0L;
	}
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Min(int attr_idx, bool& error)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		error = false;
		return _pck_min[attr_idx];
	}
	else
	{
		error = true;
		return 0.0f;
	}
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Max(int attr_idx, bool& error)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		error = false;
		return _pck_max[attr_idx];
	}
	else
	{
		error = true;
		return 0.0f;
	}
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Range(int attr_idx, bool& error)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
	{
		error = false;
		return _pck_range[attr_idx];
	}
	else
	{
		error = true;
		return 0.0f;
	}
}

float Elastic_Modeling_Job::Get_IsoOrEarth_Model_Attribute_Min(int attr_idx, bool isosphere)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
        {
		return isosphere ? _pck_iso[attr_idx] : _pck_min[attr_idx];
	}
	else
	{
		return 0.0f;
	}
}

float Elastic_Modeling_Job::Get_IsoOrEarth_Model_Attribute_Range(int attr_idx, bool isosphere)
{
	if (attr_idx >= 0 && attr_idx < _num_em_props)
        {
		return isosphere ? _pck_iso[attr_idx] : _pck_range[attr_idx];
	}
	else
	{
		return 0.0f;
	}
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute(int attr_idx, int ix, int iy, int iz)
{
	bool error;
	return Get_Earth_Model_Attribute(attr_idx,ix,iy,iz,error);
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Min(int attr_idx)
{
	bool error;
	return Get_Earth_Model_Attribute_Min(attr_idx,error);
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Max(int attr_idx)
{
	bool error;
	return Get_Earth_Model_Attribute_Max(attr_idx,error);
}

float Elastic_Modeling_Job::Get_Earth_Model_Attribute_Range(int attr_idx)
{
	bool error;
	return Get_Earth_Model_Attribute_Range(attr_idx,error);
}

int Elastic_Modeling_Job::Get_Number_Of_GPU_Pipes()
{
	return _GPU_Pipes;
}

void Elastic_Modeling_Job::Set_Number_Of_GPU_Pipes(int num_pipes)
{
	_GPU_Pipes = num_pipes;
}

int Elastic_Modeling_Job::Get_Steps_Per_GPU()
{
	return _Steps_Per_GPU;
}

void Elastic_Modeling_Job::Set_Steps_Per_GPU(int num_timesteps)
{
	_Steps_Per_GPU = num_timesteps;
}

void Elastic_Modeling_Job::Set_GPU_Devices(const int* device_ids, int num_devices)
{
	delete [] _GPU_Devices;
	_GPU_Devices = 0L;
	_num_GPU_Devices = 0;
	if (num_devices > 0)
	{
		_num_GPU_Devices = num_devices;
		_GPU_Devices = new int[num_devices];
		for (int i = 0;  i < num_devices;  ++i) _GPU_Devices[i] = device_ids[i];
	}
}

const int* Elastic_Modeling_Job::Get_GPU_Devices()
{
	return _GPU_Devices;
}

int Elastic_Modeling_Job::Get_Number_Of_GPU_Devices()
{
	return _num_GPU_Devices;
}

bool Elastic_Modeling_Job::_Calculate_ABC_Sponge(
	const char* name,
	const char* parmfile_path,
	int line_num,
	double abc_size,
	char* abc_unit,
	char* abc_flag,
	int dim,
	double cell_size,
	int& nabc_size,
	bool& nabc_flag
	)
{
	if (abc_flag != 0L)
	{
		_tolower(abc_flag);
		if (strcmp(abc_flag, "extend") == 0)
		{
			nabc_flag = true;
		}
		else
		{
			printf("%s (line %d) : Error - (Optional) %s flag must be EXTEND (you provided %s). Turn off volume extension by removing flag.\n",parmfile_path,line_num,name,abc_flag);
			return true;
		}
	}
	_tolower(abc_unit);
	if (strcmp(abc_unit, "local") == 0)
	{
		nabc_size = (int)round(ceil(abc_size/cell_size));
	}
	else if (strcmp(abc_unit, "index") == 0)
	{
		nabc_size = (int)round(abc_size);
	}
	else if (strcmp(abc_unit, "%") == 0)
	{
		nabc_size = (int)round(ceil((double)(dim-1)*abc_size/100.0));
	}
	else
	{
		printf("%s (line %d) : Error - %s unrecognized unit (%s), must be one of local, index or %%.\n",parmfile_path,line_num,name,abc_unit);
		return true;
	}
	if (_log_level > 3) printf("%s set to %d%s.\n",name,nabc_size,nabc_flag?" (EXTEND)":"");
	return false;
}

bool Elastic_Modeling_Job::_Calculate_Sub_Volume(
		const char* name,
		const char* parmfile_path,
		int line_num,
		int dim,
		double cell_size,
		double sub_min,
		double sub_max,
		char* sub_unit,
		int& ilu0,
		int& ilu1
		)
{
	_tolower(sub_unit);
	if (strcmp(sub_unit, "%") == 0)
	{
		ilu0 = (int)round(floor((double)(dim-1)*sub_min/100.0));
		ilu1 = (int)round(ceil((double)(dim-1)*sub_max/100.0));
	}
	else if (strcmp(sub_unit, "local") == 0)
	{
		ilu0 = (int)round(floor(sub_min/cell_size));
		ilu1 = (int)round(ceil(sub_max/cell_size));
	}
	else if (strcmp(sub_unit, "index") == 0)
	{
		ilu0 = (int)round(floor(sub_min));
		ilu1 = (int)round(ceil(sub_max));
	}
	if (ilu0 > ilu1)
	{
		printf("%s (line %d) : Error - %s low range is larger than high range.\n",parmfile_path,line_num,name);
		return true;
	}
	/*
	if ((ilu0 < 0 && ilu1 < 0) || (ilu1 >= dim && ilu0 >= dim))
	{
		printf("%s (line %d) : Error - %s sub volume is outside legal range.\n",parmfile_path,line_num,name);
		return true;
	}
	if (ilu0 < 0)
	{
		if (_log_level > 2) printf("%s (line %d) : Warning - %s low range clipped (%d->%d).\n",parmfile_path,line_num,ilu0,0,name);
		ilu0 = 0;
	}
	if (ilu1 >= dim)
	{
		if (_log_level > 2) printf("%s (line %d) : Warning - %s high range clipped (%d->%d).\n",parmfile_path,line_num,ilu1,dim-1,name);
		ilu1 = dim - 1;
	}
	*/
	if (_log_level > 3) printf("%s : Sub volume is [%d,%d]\n",name,ilu0,ilu1);
	return false;
}

char* Elastic_Modeling_Job::_tolower(char* str)
{
	for (int i = 0;  i < 4096 && str[i] != 0;  ++i) str[i] = tolower(str[i]);
	return str;
}

bool Elastic_Modeling_Job::_Check_Property(
		const char* prop_name,
		Voxet_Property* prop,
		double prop_val,
		size_t expected_file_size
		)
{
	if (prop != 0L)
	{
		struct stat fs;
		if (stat(prop->Get_Full_Path(), &fs) == 0)
		{
			if (fs.st_size != expected_file_size)
			{
				printf("Property %s : Error - File %s is the wrong size (%ld, expected %ld)\n",prop_name,prop->Get_Full_Path(),fs.st_size,expected_file_size);
				return true;
			}
			else
			{
				if (_log_level > 3) printf("Property %s read from file %s.\n",prop_name,prop->Get_Full_Path());
				return false;
			}
		}
		else
		{
			printf("Property %s : Error - File %s does not exist or is not readable.\n",prop_name,prop->Get_Full_Path());
			return true;
		}
	}
	else
	{
		if (_log_level > 3) printf("Property %s set to %lf.\n",prop_val);
		return false;
	}
}

Elastic_Modeling_Job::~Elastic_Modeling_Job()
{
	if (_shots != 0L)
	{
		for (int i = 0;  i < _num_shots;  ++i) delete _shots[i];
		delete [] _shots;
	}
	if (_voxet != 0L) delete _voxet;
	for (int i = 0;  i < 14;  ++i) if (_pck_moniker[i] != 0L) free(_pck_moniker[i]);
	delete [] _pck_moniker;
	delete [] _pck_mask;
	delete [] _pck_shft;
	delete [] _pck_widx;
	delete [] _pck_min;
	delete [] _pck_max;
	delete [] _pck_range;
	delete [] _GPU_Devices;
}

void Elastic_Modeling_Job::_swap_endian(float* v)
{
        int* iv = (int*)v;
        *iv =
                (((*iv) >> 24) & 255) |
                (((*iv) >>  8) & 65280) |
                (((*iv) & 65280) <<  8) |
                (((*iv) & 255) << 24);
}

void Elastic_Modeling_Job::_Read_Earth_Model(Elastic_Propagator* propagator)
{
	printf("Reading earth model...\n");
	Global_Coordinate_System* gcs = _voxet->Get_Global_Coordinate_System();

	int ilu0, ilu1, ilv0, ilv1, ilw0, ilw1;
	gcs->Convert_Transposed_Index_To_Local_Index(_sub_ix0,_sub_iy0,_sub_iz0,ilu0,ilv0,ilw0);
	gcs->Convert_Transposed_Index_To_Local_Index(_sub_ix1,_sub_iy1,_sub_iz1,ilu1,ilv1,ilw1);
	if (_log_level > 3) printf("ilu=[%d,%d] ilv=[%d,%d] ilw=[%d,%d]\n",ilu0,ilu1,ilv0,ilv1,ilw0,ilw1);

	long nw = (long)(ilw1 - ilw0 + 1);
	long nv = (long)(ilv1 - ilv0 + 1);
	long nn = nw * nv;

	long ilu = (long)ilu0;
	long nu = (long)(ilu1 - ilu0 + 1);

	long one_v_size_f = (long)(gcs->Get_NU());
	long one_w_size_f = one_v_size_f * (long)(gcs->Get_NV());

	int nthreads = 0;
#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	if (_log_level > 3) printf("Using %d thread(s).\n",nthreads);

	float* vals = new float[nu*nthreads*100];
	unsigned int** words = new unsigned int*[4];
	for (int k = 0;  k < 4;  ++k) words[k] = new unsigned int[nu*nthreads*100];

	long avgtop_cnt = 0, avgbot_cnt = 0;
	double acctop = 0.0, accbot = 0.0;

	struct timeval start;
	gettimeofday(&start, 0L);

	for (long trace_group = 0;  trace_group < nn;  trace_group+=nthreads*100)
	{
		long max_trace = trace_group + nthreads*100;
		if (max_trace > nn) max_trace = nn;

		struct timeval end;
		gettimeofday(&end, 0L);
		double elapsed = (double)end.tv_sec + (double)end.tv_usec * 1e-6 - (double)start.tv_sec - (double)start.tv_usec * 1e-6;
		if (elapsed > 1.0)
		{
			start = end;
			printf("\r%.2f%%",100.0*(double)trace_group/(double)(nn-1));
			fflush(stdout);
		}

		for (int k = 0;  k < nu*nthreads*100;  ++k)
		{
			words[0][k] = 0;
			words[1][k] = 0;
			words[2][k] = 0;
			words[3][k] = 0;
		}
		for (int attr_idx = 0;  attr_idx < _num_em_props;  ++attr_idx)
		{
			if (_props[attr_idx] != 0L)
			{
				// read traces into buffer.
				// this is done by single thread to ensure sequential read.
				FILE* fp = fopen(_props[attr_idx]->Get_Full_Path(), "rb");
				if (fp != 0L)
				{
					for (long trace = trace_group;  trace < max_trace;  ++trace)
					{
						long ilw = trace / nv;
						long ilv = trace - ilw * nv;
						ilw += (long)ilw0;
						ilv += (long)ilv0;
						long vals_off = (trace - trace_group) * nu;
						long file_off = ilw*one_w_size_f + ilv*one_v_size_f + ilu;
						//printf("_read :: file_off=%ld, vals_off=%ld, ilu=%ld, nu=%ld, ilv=%ld, ilw=%ld, trace-trace_group=%ld\n",file_off,vals_off,ilu,nu,ilv,ilw,trace-trace_group);
						if ((file_off % one_v_size_f) != 0) {printf("file_off = %ld\n",file_off); exit(0);}
						fseek(fp, file_off*sizeof(float), SEEK_SET);
						long nread = fread(vals+vals_off, sizeof(float), nu, fp);
						if (nread != nu) printf("_read :: offset=%ld, ilu=%ld, ilv=%ld, ilw=%ld -- tried to read %ld, got %ld\n",file_off,ilu,ilv,ilw,nu,nread);
					}
					int fid = fileno(fp);
					fdatasync(fid);
					posix_fadvise(fid,0,0,POSIX_FADV_DONTNEED);
					fclose(fp);
				}
				else
				{
					printf("ERROR! Failed to open %s for reading.\n",_props[attr_idx]->Get_Full_Path());
					exit(-1);
				}
			}
				
			// compress and store
#pragma omp parallel for reduction(+:acctop,accbot,avgtop_cnt,avgbot_cnt)
			for (long trace = trace_group;  trace < max_trace;  ++trace)
			{
				long ilw = trace / nv;
				long ilv = trace - ilw * nv;
				ilw += (long)ilw0;
				ilv += (long)ilv0;
				long vals_off = (trace - trace_group) * nu;
				for (int sample = 0;  sample < nu;  ++sample)
				{
					if (_props[attr_idx] != 0L)
					{
						_swap_endian(&(vals[vals_off+sample]));
						//if (attr_idx == Attr_Idx_Vp && sample < 2) printf("vals[%d+%d] = %f\n",vals_off,sample,val);
					}
					else
					{
						vals[vals_off+sample] = _const_vals[attr_idx];
					}
					if (attr_idx == Attr_Idx_Q) vals[vals_off+sample] = 1.0f / vals[vals_off+sample];
					_Pack_Earth_Model_Attribute(words[_pck_widx[attr_idx]][vals_off+sample],attr_idx,vals[vals_off+sample]);
				}
				if (attr_idx == Attr_Idx_Vp)
				{
					if (gcs->U_Is_Z())
					{
						acctop += vals[vals_off];	++avgtop_cnt;
						accbot += vals[vals_off+nu-1];	++avgbot_cnt;
					}
					else
					{
						if ((gcs->V_Is_Z() && ilv == ilv0) || (gcs->W_Is_Z() && ilw == ilw0)) // zztop
						{
							for (int k = 0;  k < nu;  ++k) acctop += vals[vals_off+k];
							avgtop_cnt += nu;
						}
						else if ((gcs->V_Is_Z() && ilv == ilv1) || (gcs->W_Is_Z() && ilw == ilw1)) // zzbot
						{
							for (int k = 0;  k < nu;  ++k) accbot += vals[vals_off+k];
							avgbot_cnt += nu;
						}
					}
				}
			}
		}

#pragma omp parallel for
		for (long trace = trace_group;  trace < max_trace;  ++trace)
		{
			long ilw = trace / nv;
			long ilv = trace - ilw * nv;
			ilw += (long)ilw0;
			ilv += (long)ilv0;
			long vals_off = (trace - trace_group) * nu;

			int x0, x1, y0, y1, z0, z1;
			gcs->Convert_Local_Index_To_Transposed_Index(ilu0,ilv,ilw,x0,y0,z0);
			gcs->Convert_Local_Index_To_Transposed_Index(ilu1,ilv,ilw,x1,y1,z1);
			int xinc = x0 < x1 ? 1 : 0;
			int yinc = y0 < y1 ? 1 : 0;
			int zinc = z0 < z1 ? 1 : 0;
			x0 = x0 - _prop_x0;
			y0 = y0 - _prop_y0;
			z0 = z0 - _prop_z0;
			//printf("il-u-v-w[%d,%d][%d][%d] => x0,y0,z0=%d,%d,%d xinc,yinc,zinc=%d,%d,%d\n",ilu0,ilu1,ilv,ilw,x0,y0,z0,xinc,yinc,zinc);
			propagator->_Insert_Earth_Model_Stripe(
					&(words[0][vals_off]),
					&(words[1][vals_off]),
					&(words[2][vals_off]),
					&(words[3][vals_off]),
					nu,
					x0,xinc,y0,yinc,z0,zinc
					);
		}
	}
	_vpvert_avgtop = (float)(acctop / (double)avgtop_cnt);
	_vpvert_avgbot = (float)(accbot / (double)avgbot_cnt);
	printf("\navg(Vp@top)=%f - avg(Vp@bot)=%f\n",_vpvert_avgtop,_vpvert_avgbot);

	delete [] vals;
	for (int k = 0;  k < 4;  ++k) delete [] words[k];
	delete [] words;
	
	propagator->_NABC_TOP_Extend(_sub_iz0 - _prop_z0);
	propagator->_NABC_BOT_Extend(_sub_iz1 - _prop_z0);
	propagator->_NABC_SDX_Extend(_sub_ix0 - _prop_x0, _sub_ix1 - _prop_x0);
	propagator->_NABC_SDY_Extend(_sub_iy0 - _prop_y0, _sub_iy1 - _prop_y0);
}

