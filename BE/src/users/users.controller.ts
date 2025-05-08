import {
  Body,
  Controller,
  Get,
  Param,
  Patch,
  Query,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { CurrentUser } from 'src/auth/@decorator/current-user.decorator';
import { Roles } from 'src/auth/@decorator/roles.decorator';
import { ROLE } from 'src/auth/util/@enum/role.enum';
import { CloudinaryService } from 'src/file/file.service';
import { GetUserByKeywordDTO } from './dto/get-user.dto';
import { UpdateUserDto } from './dto/update-user.dto';
import { UsersService } from './users.service';

@Controller('users')
export class UsersController {
  constructor(
    private usersService: UsersService,
    private cloudinaryService: CloudinaryService,
  ) {}

  @Get('/doctor/:doctorId')
  findAll(@Param('doctorId') doctorId: string) {
    return this.usersService.findDoctorByID(doctorId);
  }

  @Get('/doctors')
  findAllDoctors(@CurrentUser() currentUser: any) {
    return this.usersService.findAllDoctors();
  }

  @Get('/keyword')
  async findUserByKeyword(@Body() keyword: GetUserByKeywordDTO) {
    return await this.usersService.findUserByKeyword(keyword);
  }

  @Patch('/edit-profile')
  async editProfile(
    @CurrentUser('userId') userId: string,
    @Body() updateUserDto: UpdateUserDto,
  ) {
    return await this.usersService.editProfile({ ...updateUserDto }, userId);
  }

  @Patch('change/avatar')
  @UseInterceptors(FileInterceptor('file'))
  async updateAvatar(
    @CurrentUser('userId') userId: string,
    @UploadedFile() file: Express.Multer.File,
  ) {
    const uploadedFile = await this.cloudinaryService.uploadFile(file);
    return await this.usersService.updateAvatar(userId, uploadedFile.url);
  }
}
